# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
import logging
import pathlib
import pickle
import uuid
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from glob import glob

import dask
import dill

from .static import StaticExperimentTree

logger = logging.getLogger("dagger")


class Experiment:
    """
    An Experiment object retains state for all the ExperimentStates and
    subsequent transitions that are governed by Recipe objects. This is used
    by dask to store and then execute the many computation paths that give
    origin to the various experiment states in the experiment.

    Attributes:
        directory (str | pathlib.Path): The directory all the stages of the
        experiment will be stored in.
        leaves (Dict[uuid.UUID, dask.Delayed]): (internal) Stores the final
            states for leaf nodes in the Experiment graph. Dask will traverse
            until the leaf nodes are computed, in the case where we lazily
            define the graph.
        nodes (Dict[uuid.UUID, dask.Delayed]): (internal) Stores the final
            states for all nodes in the Experiment graph.
        root (ExperimentState): The (never changing) root of the
            Experiment graph.
        state_class: python class of which all experiment states in this
            experiment are instances.
        experiment_arguments (dict): information and properties that
            define the experiment and its root.
        graph (StaticExperimentTree): bookeeping object to access all
            topologically-sorted nodes in the experiment.
        tags (list): tags associated with experiment states in this experiment
    """

    def __init__(self, directory, state_class):
        """Create a new Experiment object.

        An Experiment provides a sort-of anchor for a DAG of experimental steps.
        To create an Experiment, you must provide a directory, which will be the
        location for the Experiment to deposit a file for each state in the
        experiment graph.

        Args:
            directory (str | pathlib.Path): Path to a directory (will be
                created) which will be the root for all experment state
                transitions to be stored.
            state_class: python class of which all experiment states in this
                experiment are instances.
        """
        self.nodes = {}
        self.leaves = {}
        self.root = None
        self.directory = pathlib.Path(directory).absolute()
        self.directory.mkdir(exist_ok=True)
        self.state_class = state_class
        self.experiment_arguments = None

        # The graph is only built up when loading an already-run graph
        self.graph = None

        # This is a tag manager
        self.tags = []

    @contextmanager
    def tag(self, tag):
        """Context manager that associates each state defined in this context
        with the `tag` passed in as an argument. Extends the experiment's tags
        list while in the context.
        """
        if isinstance(tag, str):
            tag = [tag]
        original_tags = self.tags[:]
        self.tags.extend(tag)
        yield
        self.tags = original_tags

    def spawn_new_tree(self, **experiment_arguments):
        """
        Create the initial experiment state to anchor off of the current
        Experiment object.

        Args:
            experiment_arguments (dict): information and properties that
            define the experiment and its root.

        Returns:
            ExperimentStatePromise: The initial state represented as a
                promise, so not yet computed or realized.

        Raises:
            RuntimeError: If the Experiment object already has a root
                ExperimentState.

        """
        if self.root:
            raise RuntimeError(
                "This Experiment object already has a root ExperimentState!"
            )

        self.experiment_arguments = experiment_arguments

        # Here, we begin defining what should be contained in an
        # ExperimentState. The main thing to note here is that an
        # ExperimentState is initialized with a parent_sha (an identifier of
        # its progenitor state) and the experiment object itself. This is
        # critical so every ExperimentState in an Experiment knows where it
        # comes from and can find its root.
        initial_state = self.state_class(
            parent_sha=None, experiment_object=self
        )
        initial_state.root_state = True

        for prop in getattr(initial_state.__class__, "PROPERTIES", []):
            # Will raise a KeyError if not passed in
            if prop not in experiment_arguments:
                raise KeyError(
                    "State property {} not in found".format(prop)
                    + "in `experiment_arguments`"
                )
            setattr(initial_state, prop, experiment_arguments[prop])

        # Needs to be implemented when subclassing ExperimentState
        initial_state.initialize_state(**experiment_arguments)

        if not initial_state.restore():
            logger.info("Root state unsaved - saving now")
            initial_state.save()

        self.root = initial_state
        del initial_state
        # Turn the root into a state promise to attach to the dask tree
        return ExperimentStatePromise(
            promise=self.root,
            id=uuid.uuid4(),
            previous_id=None,
            experiment_object=self,
        )

    def save(self):
        """Save minimal information needed to start reconstructing the
        experiment, i.e. how to locate the root, as well as other basic info
        about the experiment parameters. These are saved in human-readable
        json in <directory>/experiment.json
        """
        json.dump(
            {
                "root": self.root.sha(),
                "experiment_arguments": self.experiment_arguments,
            },
            open(self.directory / "experiment.json", "w"),
            indent=4,
        )

    @classmethod
    def restore(cls, directory, state_class=None, slim=True):
        """Restoring an experiment means reinstantiating an Experiment
        object with all correct attributes previously stored in the `save`
        phase using json. This will load the slim version of all experiments
        found in the `directory`, and use their slim info (found in the
        <sha>.slim.json files in the `directory`) to reconstruct the `edge_map`
        and `node_map`, thus reconstructing the whole experimental tree
        structure and storing it in a `StaticExperimentTree` for analysis.

        Args:
            directory (str | pathlib.Path): Path to a directory (will be
                created) which will be the root for all experment state
                transitions to be stored.
            state_class: python class of which all experiment states in this
                experiment are instances.
            slim (bool): whether to lead the ExperimentStates in slim format
                when restoring the experiment (True) or in full format (False)

        Returns:
            The loaded Experiment corresponding to all states found in the
                specified `directory`.
        """
        state_class = state_class or ExperimentState
        directory = pathlib.Path(directory)
        experiment = cls(directory, state_class)

        experiment.__dict__.update(
            json.load(open(directory / "experiment.json", "r"))
        )

        # .root is first loaded in as a string sha.
        experiment.root = state_class.load(
            experiment.root, experiment=experiment, slim=slim
        )

        # Get all experiments by looking for all .slim.json files in the
        # directory
        all_shas = [
            pathlib.Path(p).name.replace(".slim.json", "")
            for p in glob(f"{directory / '*.slim.json'}")
        ]

        # Reconstruct the experiment tree by connecting all nodes with edges
        # using the parent and children info contained in each state
        edge_map = defaultdict(set)
        node_map = dict()

        for sha in all_shas:
            if sha != experiment.root.sha():
                state = state_class.load(
                    sha=sha, experiment=experiment, slim=slim
                )
            else:
                state = experiment.root
            node_map[sha] = state
            if state.parent_sha:
                edge_map[state.parent_sha].add(sha)

        # Run a topological sort so we can filter states by distance from the
        # root
        toposort = defaultdict(set)

        queue = {experiment.root.sha()}
        depth = 0

        while queue:
            new_queue = set()
            for node in queue:
                toposort[depth].add(node)
                for child_node in edge_map[node]:
                    new_queue.add(child_node)
            depth += 1
            queue = new_queue

        # Make sure we can traverse the graph sensically
        while depth > 0:
            for node in map(node_map.get, toposort.get(depth, [])):
                if node:
                    parent = node_map.get(node.parent_sha)
                    if not hasattr(parent, "children"):
                        parent.children = {node}
                    else:
                        parent.children.add(node)
                    node.parent = parent
            depth -= 1

        experiment.root.parent = None
        # Represent the set of connected nodes and edges as
        # a StaticExperimentTree
        experiment.graph = StaticExperimentTree(node_map, edge_map, toposort)

        return experiment

    def run(self, scheduler="single-threaded"):
        """Run the Experiment defined by the graph using dask.

        Args:
            scheduler (str): How dask should schedule the nodes in the graph
                (see the dask documentation for more information).
        """
        self.save()  # save basic experiment info to json
        _ = dask.compute(self.leaves, scheduler=scheduler)
        # when dask goes thru the tree, it knows the full sequence of ops
        # needed to compute each leaf, so this gives dask full authority in
        # determining the best dispatch path.


class ExperimentStatePromise:
    """An ExperimentStatePromise is a construct to allow for a lazy graph.

    Specifically, an ExperimentStatePromise encapsulates a lazy-evaluated
    function that represents the transition from state to state.

    Attributes:
        promise (object): The object to be lazily acted upon.
        id (uuid.UUID): Unique ID of this experiment state promise.
        previous_id (uuid.UUID): ID of the state which directly feeds into the
            next state defined by the realization of the promise.
        experiment_object (Experiment): The anchor Experiment for this promise.
    """

    def __init__(self, promise, id, previous_id, experiment_object):
        """Creates a new promose from an existing object.

        Args:
            promise (object): The object to be lazily acted upon.
            id (uuid.UUID): Unique ID of this experiment state promise.
            previous_id (uuid.UUID): ID of the state which directly feeds
                into the next state defined by the realization of the promise.
            experiment_object (Experiment): The anchor Experiment for
                this promise.
        """
        self.promise = promise
        self.id = id
        self.previous_id = previous_id
        self.experiment_object = experiment_object

    def promise_from_callable(self, fn):
        """Defines the function which is to act on the promise which evolves
        the object from state A to state B.

        Args:
            fn (Callable): Function that takes as input an object of whatever
                type the promise is and returns a python object.

        Returns:
            ExperimentStatePromise
        """
        return ExperimentStatePromise(
            promise=dask.delayed(fn)(self.promise),
            id=uuid.uuid4(),
            previous_id=self.id,
            experiment_object=self.experiment_object,
        )

    def get(self, scheduler="single-threaded"):
        """Evaluate the graph to turn the current state from a promise into
        a computed experiment state.

        Args:
            scheduler (str): How dask should schedule the nodes in the graph
                (see the dask documentation for more information).
        """
        # this is for all the lazily evaluated states
        if hasattr(self.promise, "compute"):
            return self.promise.compute(scheduler=scheduler)
        # if a state is already materialized
        else:
            return self.promise


class ExperimentState:
    """An ExperimentState represents a point in time with the evolution of an
    experiment. An ExperimentState can have any number of objects attached to
    it as attributes. Importantly, we must have a consistent way to save a
    state such that it can be reloaded if an identical experiment is to be
    run (cacheing).

    Attributes:
        experiment_object (Experiment): The anchor experiment for the state.
        parent_sha (str): Path to the saved version of the parent state.
        from_chache (bool): TODO
        root_state (bool): whether this experiment state is the root.
        save_pre_hooks (list):
        save_post_hooks (list):
        load_post_hooks (list):
        recipe (Recipe): recipe that generated the current state. None if
            this is the root.
        slim_loaded (bool): whether the state is only loaded in its slim
            version (True), or in its full version (False)
        slim_sha (str): sha generated to uniquely identify this state
        tags (list): strings that identify the state in the graph
        parent (ExperimentState): state containing the direct parent of the
            current ExperimentState. None if this is the root.
        children (set): ExperimentState objects that are directly derived
            from the current state
        root (ExperimentState): state containing the root of the experiment the
            current ExperimentState belongs to.
        directory (PosixPath): directory where the experiment is stored on
            disk
        path (PosixPath): path to the current ExperimentState on disk

    """

    PROPERTIES = []  # will be stored
    NONHASHED_ATTRIBUTES = []  # will not be stored

    def __init__(self, parent_sha=None, experiment_object=None):

        self.parent_sha = parent_sha
        self.experiment_object = experiment_object
        self.from_cache = False
        self.root_state = False

        # Set up pre- and post-hooks for saving and loading.
        self.save_pre_hooks = []
        self.save_post_hooks = []
        # N.B., the concept of a load-pre-hook doesnt really make sense.
        self.load_post_hooks = []

        # Represent the recipe that describes the state transition leading
        # to the current state. This allows us to cache *just* the nodes.
        self._recipe = None

        # Identify whether this is a slim-loaded state.
        self.slim_loaded = False
        self.slim_sha = None

        # Keep a set of tags which help you identify the node in the graph
        self.tags = []

        # Initialize all properties to None
        for prop in self.PROPERTIES + self.NONHASHED_ATTRIBUTES:
            setattr(self, prop, None)

        self.parent = None
        self.children = set()
        # If the experiment state has a parent, the following attributes will
        # be inherited. These are shared across all states that stem from the
        # same root
        if self.parent_sha is not None:
            parent = self.__class__.load(self.parent_sha, experiment_object)
            for property_name in (
                self.__class__.PROPERTIES
                + self.__class__.NONHASHED_ATTRIBUTES
            ):
                value = getattr(parent, property_name, None)
                setattr(self, property_name, value)

            # remove parent -- no longer needed
            parent.deflate()
            del parent

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(sha={self.sha()}, tags={self.tags})"
        )

    @property
    def recipe(self):
        return self._recipe

    @property
    def root(self):
        if self.experiment_object is not None:
            return self.experiment_object.root

    def sha(self):
        """Computes the unique hash value associated with this state"""

        if self.slim_loaded:
            return self.slim_sha
        obj_keys = [
            "parent_sha",
            "_recipe",
            "root_state",
            "tags",
        ] + self.PROPERTIES

        def marshal_obj(obj):
            if isinstance(obj, str):
                return obj
            if isinstance(obj, pathlib.Path):
                return str(obj)
            if isinstance(obj, dict):
                return str(
                    sorted(
                        list(
                            (marshal_obj(k), marshal_obj(v))
                            for k, v in obj.items()
                        )
                    )
                )

            if isinstance(obj, Recipe):
                return str(
                    sorted(
                        list(
                            (
                                marshal_obj(prop),
                                marshal_obj(getattr(obj, prop, None)),
                            )
                            for prop in getattr(obj, "PROPERTIES", [])
                        )
                    )
                )

            if isinstance(obj, (list, tuple)):
                return str(list(marshal_obj(o) for o in obj))

            if isinstance(obj, set):
                return str(sorted(list(marshal_obj(o) for o in obj)))

            if isinstance(obj, int):
                return str(obj)

            if isinstance(obj, float):
                return f"{obj:.8f}"

            return pickle.dumps(obj)

        obj = {
            prop: marshal_obj(getattr(self, prop, None)) for prop in obj_keys
        }

        representation = str(sorted(list(obj.items())))
        h = hashlib.md5()
        h.update(representation.encode())
        return h.hexdigest() + ("-root" if self.root_state else "")

    def __getstate__(self):
        """
        The getstate hook is invoked by dill and pickle - we don't want to
        serialize the experiment object!
        """
        o = dict(self.__dict__)
        del o["experiment_object"]
        return o

    def __setstate__(self, s):
        """
        The setstate hook is invoked by dill and pickle - we don't want to
        serialize the experiment object!
        """
        self.__dict__ = s
        self.experiment_object = None

    def restore(self):
        """Fully load a state back into memory.

        Returns:
            True, if the state has been inflated back in memory in its full
            version; False, if the state wasn't found and wasn't loaded back
            up correctly.
        """
        path = self.path
        parent = self.parent
        if not hasattr(self, "children"):
            self.children = set()
        children = set(self.children)

        self.slim_loaded = False
        self.slim_sha = None

        if path.exists():
            experiment_object = self.experiment_object
            logger.info(
                f"State already exists at {path}! "
                "Safely loading existing state."
            )

            # Reset exp object because it gets deleted from state before
            # serialization
            self.__dict__ = self.__class__.load(
                path, experiment_object
            ).__dict__

            self.from_cache = True
            self.experiment_object = experiment_object
            self.parent = parent
            self.children = children

            return True
        logger.info(f"No cached state at: {path}")
        return False

    def deflate(self):
        """Reduce the state back to its "slim" version by deleting all its
        attributes that are not in ["parent_sha", "slim_sha", "slim_loaded",
        "tags", "experiment_object", "parent", "children"]. This helps reduce
        the memory footprint.

        Returns:
            True at the end of all defation operations
        """
        sha = self.sha()

        for slot in list(self.__dict__.keys()):
            if slot not in {
                "parent_sha",
                "slim_sha",
                "slim_loaded",
                "tags",
                "experiment_object",
                "parent",
                "children",
            }:
                o = getattr(self, slot)
                delattr(self, slot)
                del o

        self.slim_loaded = True
        self.slim_sha = sha
        return True

    @contextmanager
    def lazy_load(self):
        """Returns the restored version of the state and automatically
        handles deflating it when no longer in scope.
        """
        try:
            self.restore()
            yield
        finally:
            self.deflate()

    @property
    def directory(self):
        base = "./"
        if self.experiment_object:
            base = self.experiment_object.directory
        return pathlib.Path(base).absolute()

    @property
    def path(self):
        """Path of this state on disk, obrained from the Experiment's
        directory and the state's sha."""
        return self.directory / self.sha()

    def new_state(self, recipe=None):
        state = self.__class__(
            parent_sha=self.sha(), experiment_object=self.experiment_object
        )
        state._recipe = recipe
        # TODO: should we add state to self.children??
        # TODO: should we add self to state.parent??
        return state

    def save(self):
        """Serializes the state by dumping the json version of it after
        executing all saving pre-hooks and hooks. A slim representation of the
        state will also be saved by appending .slim.json to the file name for
        quick experiment reconstruction and state reloading.
        """
        path = self.path
        sha = self.sha()
        logger.debug(f"Saving to: {path}")

        for hook in self.save_pre_hooks:
            hook()

        if hasattr(self, "save_hook") and callable(
            getattr(self, "save_hook")
        ):
            logger.debug("using custom save_hook")
            getattr(self, "save_hook")(path)
        else:
            dill.dump(self, open(path, "wb"))

        for hook in self.save_post_hooks:
            hook()

        slim_repr = {
            "parent_sha": self.parent_sha,
            "slim_sha": sha,
            "slim_loaded": True,
            "tags": self.tags,
        }

        json.dump(slim_repr, open(f"{path}.slim.json", "w"), indent=4)

    @classmethod
    def load(cls, sha, experiment=None, slim=False):
        """Reloads the state identified by `sha` into memory, either in its
        slim format (slim=True) or its full format after executing loading
        hooks (slim=False).

        Args:
            sha (str): sha generated to uniquely identify this state
            experiment (Experiment): object that contains this node
            slim (bool):whether to lead the ExperimentState in slim format
                (True) or in full format (False)

        Returns:
            A reloaded ExperimentState
        """
        if isinstance(experiment, (str, pathlib.Path)):
            experiment_object = Experiment(
                pathlib.Path(experiment).absolute(), cls
            )
        else:
            experiment_object = experiment or Experiment(
                pathlib.Path(".").absolute(), cls
            )

        path = pathlib.Path(experiment_object.directory) / sha

        if slim:
            state_dict = json.load(open(f"{path}.slim.json", "r"))
            state = cls.__new__(cls)
            state.__setstate__(state_dict)
        else:
            if hasattr(cls, "load_hook") and callable(
                getattr(cls, "load_hook")
            ):
                logger.debug("using custom load_hook")
                state = getattr(cls, "load_hook")(path)
            else:
                state = dill.load(open(path, "rb"))

            for hook in state.load_post_hooks:
                hook()

        state.experiment_object = experiment_object
        return state

    def to_promise(self):
        """Turn the ExperimentState into an ExperimentStatePromise to be able
        to add another node to modify the graph on the fly.

        Returns:
            An ExperimentStatePromise corresponding to the current
            ExperimentState.
        """
        return ExperimentStatePromise(
            promise=self,
            id=uuid.uuid4(),
            previous_id=None,
            experiment_object=self.experiment_object,
        )


class Recipe:
    """A Recipe represents a sequence of actions that modify an
    ExperimentState. Subclass it and implement the `run` method to define
    how the Recipe transforms an experiment state into another one.
    """

    def __call__(self, experiment_state):
        """This is what adds the ops defined by the recipe to the graph
        """
        new_state = experiment_state.promise_from_callable(
            partial(
                self.run_recipe,
                # We want to "dereference" the tag list, as we're not
                # guaranteed execution order and byref will lead to race
                # conditions in execution plan.
                tags=experiment_state.experiment_object.tags[:],
            )
        )

        # If the previous state (`experiment_state`) was a leaf, but we now
        # created a new state stemming from it, then the previous state can
        # be removed from the set of leaves.
        if experiment_state.id in experiment_state.experiment_object.leaves:
            del experiment_state.experiment_object.leaves[experiment_state.id]

        # The new state can now be added to the leaves
        new_state.experiment_object.leaves[new_state.id] = new_state.promise
        # Add it to the nodes as well
        new_state.experiment_object.nodes[new_state.id] = new_state.promise
        return new_state

    def run_recipe(self, prev_state, tags=None):
        """Give birth to a new state from the state upon which the recipe is
        acting, then modify this new state according to the instructions in
        the recipe.
        """
        new_state = prev_state.new_state(self)
        new_state.tags = tags
        prev_state.deflate()
        new_state.restore()

        # If we aren't in a cached state, actually do the work.
        if not new_state.from_cache:
            new_state = self.run(new_state)
            if not isinstance(new_state, ExperimentState):
                raise RuntimeError(
                    f"run() method missing valid return type. "
                    f"Found {type(new_state)}, expected ExperimentState"
                )
            new_state.save()
        new_state.deflate()
        return new_state


class Function:
    """Gives the ability to execute any function on an experiment state
    without modifying the graph experiment states and adding new nodes. This
    is intended for evaluation and analysis functions that do no cause a state
    to logically transition into a new modified state. These functions should
    not be modifying the state in any way, they should simply be probing it.
    """

    def __init__(self, op):
        self.op = op

    def __call__(self, experiment_state):
        """This is what adds the ops defined by the recipe to the graph
        """

        promise = dask.delayed(self._safe_op)(experiment_state.promise)

        # If the previous state (`experiment_state`) was a leaf, but we now
        # created a new state stemming from it, then the previous state can
        # be removed from the set of leaves.
        if experiment_state.id in experiment_state.experiment_object.leaves:
            del experiment_state.experiment_object.leaves[experiment_state.id]

        # The new state can now be added to the leaves
        experiment_state.experiment_object.leaves[uuid.uuid4()] = promise

        return experiment_state

    def _safe_op(self, state):
        state.restore()
        try:
            self.op(state)
        finally:
            state.deflate()


# Makes decorator-style prettier
function = Function
