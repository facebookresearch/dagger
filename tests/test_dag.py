# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import random
from operator import add, mul

import pytest
from dask.delayed import Delayed

from dagger import (
    Experiment,
    ExperimentState,
    ExperimentStatePromise,
    Function,
    Recipe,
)


class EmptyState(ExperimentState):
    PROPERTIES = ["a", "B"]
    NONHASHED_ATTRIBUTES = ["c"]

    def initialize_state(self, **kwargs):
        pass


class OpRecipe(Recipe):
    PROPERTIES = ["op", "x", "stochastic"]

    def __init__(self, op, x, stochastic=False):
        self.op = op
        self.x = x
        self.stochastic = stochastic

    def run(self, s):
        s.c = self.op(s.a, self.x)
        if self.stochastic:
            s.c *= random.gauss(0, 1)
        return s


class TestExperimentState:

    def test_state_init(self):
        """Test that at initialization all attributes are None or empty"""
        s = EmptyState()

        assert s.parent is None
        assert s.parent_sha is None
        assert not s.children
        assert s.experiment_object is None
        assert not s.from_cache
        assert not s.root_state
        assert s.recipe is None
        assert hasattr(s, "a") and s.a is None
        assert hasattr(s, "B") and s.B is None
        assert hasattr(s, "c") and s.c is None

    def test_state_sha_identical(self):
        """Two identical states should have the same hash"""
        s = EmptyState()
        r = EmptyState()
        assert r.sha() == s.sha()

    def test_state_sha_indep_exp(self, tmpdir):
        """Test that the sha is independent of which experiment the state
        belongs to, and it only captures the nature of the state itself.
        """
        exp = Experiment(directory=tmpdir, state_class=EmptyState)

        s_lone = EmptyState(experiment_object=None)
        s = EmptyState(experiment_object=exp)

        assert s_lone.sha() == s.sha()

    def test_state_exp(self, tmpdir):
        """Test that the experiment object is correctly assigned to the state
        when specified, but is removed when calling __getstate__ (used for
        serialization) and is set to None by __setstate___.
        """
        exp = Experiment(directory=tmpdir, state_class=EmptyState)

        s = EmptyState(experiment_object=exp)
        assert "experiment_object" in s.__dict__
        assert s.experiment_object == exp
        assert "experiment_object" not in s.__getstate__()

        # Resetting the state from another state should set the experiment to
        # None because the experiment may either not be instantiated if states
        # are being reloaded from disk, or, in general, the copy of a state
        # may not logically belong to the same experiment as the original one.
        s.__setstate__(s.__dict__)
        assert "experiment_object" in s.__dict__
        assert s.experiment_object is None

    def test_state_root_inheritance(self, tmpdir):
        """Test that a state that is instatiated within an experiment that
        has a root will correctly inherit the pointer to the root to be able
        to refer back to it when needed (e.g. for weight resetting).
        """
        exp = Experiment(directory=tmpdir, state_class=EmptyState)

        root = EmptyState(experiment_object=exp)
        exp.root = root  # manually set it to the root of the experiment

        # Ideally this should also be set to True, but this is not used here
        # and root setting should never happen manually anyways, so this is
        # internally handled when a new tree is spawned.
        # root.root_state = True

        s = EmptyState(experiment_object=exp)
        assert s.root == root

    def test_state_save(self, tmpdir):
        """Test that a state created within an experiment inherits the right
        directory and gets saved in there upon calling the `save` method. This
        should generate a pickle file with the state and a json file with the
        state info for slim reloading.
        """
        exp = Experiment(directory=tmpdir, state_class=EmptyState)
        s = EmptyState(experiment_object=exp)  # assign within exp
        assert s.directory == tmpdir
        s.save()  # save state to disk
        assert os.path.isfile(os.path.join(tmpdir, s.sha()))
        assert os.path.isfile(os.path.join(tmpdir, s.sha() + ".slim.json"))

    def test_state_restore(self, tmpdir):
        # Create state and assign some properties
        exp = Experiment(directory=tmpdir, state_class=EmptyState)
        s = EmptyState(experiment_object=exp)
        s.a = 5
        s.B = "hello"
        s.c = 12345
        # Save it to disk
        s.save()
        assert not s.from_cache
        # Deflate it to remove all properties and put it in "slim" state
        s.deflate()
        assert not hasattr(s, "a")
        assert not hasattr(s, "B")
        assert not hasattr(s, "c")

        # Restore it from disk to demonstrate that it was saved correctly and
        # we are able to recover the properties
        s.restore()
        assert s.a == 5
        assert s.B == "hello"
        assert s.c == 12345
        assert s.from_cache

    def test_save_hooks(self, tmpdir):
        """Test that save pre- and post-hooks are working correctly.
        Note: this should be done with the property `c` which is part of the
        NONHASHED_ATTRIBUTES. If we modified `a` or `B`, the hash would
        change because they contribute to its value, and restoring would fail
        because no corresponding state is found on disk. See below for test of
        that behavior.
        """
        from types import MethodType

        exp = Experiment(directory=tmpdir, state_class=EmptyState)
        s = EmptyState(experiment_object=exp)
        # Set c to 5 to begin with
        s.c = 5

        def change_c(self, value):
            self.c = value

        s.change_c = MethodType(change_c, s)
        # Before saving, set c to 6
        s.save_pre_hooks = [lambda: s.change_c(6)]
        # After saving, set c to 7
        s.save_post_hooks = [lambda: s.change_c(7)]

        s.save()
        # We are after saving, so c should be 7
        assert s.c == 7

        # Deflate and reload the state from disk. The saved version of the
        # state should have the value of c that was set before saving, i.e. 6
        s.deflate()
        s.restore()
        assert s.c == 6

    def test_state_sha_save_hook(self, tmpdir):
        """When restoring fails, it returns False. Check that the save post
        hook sets the new value correctly, thus modifying the hash value of
        the state when the property that gets modified is part of PROPERTIES.
        """
        from types import MethodType

        exp = Experiment(directory=tmpdir, state_class=EmptyState)
        s = EmptyState(experiment_object=exp)
        # Set a to 5 to begin with
        s.a = 5

        def change_a(self, value):
            self.a = value

        s.change_a = MethodType(change_a, s)
        # After saving, set a to 7
        s.save_post_hooks = [lambda: s.change_a(7)]

        s.save()
        s.deflate()
        assert not s.restore()

    def test_new_state(self, tmpdir):
        """Generating a new state from a previous one using new_state should
        generate the right connection between the two states, which can be
        inspected through the setting of a parent_sha and then in the way
        the experiment graph is drawn when the StaticExperimentTree is
        reloaded.
        """
        exp = Experiment(directory=tmpdir, state_class=EmptyState)
        s = EmptyState(experiment_object=exp)
        # Set it as the root of the experiment
        exp.root = s
        s.save()

        # Generate a new child state from state `s`
        r = EmptyState.new_state(s)
        assert r.parent_sha == s.sha()
        assert r.experiment_object == exp

        r.save()
        exp.save()
        exp = Experiment.restore(directory=tmpdir)  # reload experiment

        # Test that the graph looks as expected with the connection
        assert len(exp.graph.nodes) == 2
        assert exp.graph.edge_map[s.sha()] == set([r.sha()])

    def test_state_lazy_load(self, tmpdir):
        exp = Experiment(directory=tmpdir, state_class=EmptyState)
        s = EmptyState(experiment_object=exp)
        exp.root = s
        s.save()
        # Generate a new child state from state `s`
        r = EmptyState.new_state(s)
        r.save()
        exp.save()

        # Reload the two-state experiment
        # By default, slim=True
        exp = Experiment.restore(directory=tmpdir)

        for node, state in exp.graph.node_map.items():
            assert state.slim_loaded  # deflated
            with state.lazy_load():
                assert not state.slim_loaded  # fully loaded
            assert state.slim_loaded  # deflated

        # Check behavior change as slim is set to False when exp is restored
        exp = Experiment.restore(directory=tmpdir, slim=False)

        for node, state in exp.graph.node_map.items():
            assert not state.slim_loaded  # deflated
            with state.lazy_load():
                assert not state.slim_loaded  # fully loaded
            # Note: lazy_load deflates the state even if it was initially
            # fully loaded!
            assert state.slim_loaded  # deflated


class TestExperiment:

    def test_experiment_init(self, tmpdir):
        """Test that the conditions we expect after the initialization of an
        Experiment are met."""
        exp = Experiment(directory=tmpdir, state_class=ExperimentState)
        assert os.path.isdir(tmpdir)
        assert exp.root is None

    def test_experiment_tags(self, tmpdir):
        """Test that the tags context manager is working as designed by
        adding tags to the experiment when inside the corresponding with
        statement."""
        exp = Experiment(directory=tmpdir, state_class=ExperimentState)
        assert not exp.tags

        with exp.tag("test_tag"):
            assert exp.tags == ["test_tag"]

        assert not exp.tags

        with exp.tag("test_tag"):
            with exp.tag("second_tag"):
                assert exp.tags == ["test_tag", "second_tag"]
            assert exp.tags == ["test_tag"]

    def test_spawn_new_tree_error(self, tmpdir):
        exp = Experiment(directory=tmpdir, state_class=EmptyState)
        exp_args = {}
        with pytest.raises(KeyError):
            exp.spawn_new_tree(**exp_args)

    def test_spawn_new_tree(self, tmpdir):
        """
        """
        exp = Experiment(directory=tmpdir, state_class=EmptyState)

        # The argument 'c' should be ignored, as it is not set from thew
        # constructor in spawn_new_tree
        exp_args = {"a": "first", "B": "second", "c": "shouldnotpropagate"}
        root = exp.spawn_new_tree(**exp_args)

        assert type(root) == ExperimentStatePromise

        root_state = root.get()
        assert exp.root is root_state
        assert isinstance(root_state, EmptyState)
        assert root_state.a == exp_args["a"]
        assert root_state.B == exp_args["B"]
        assert root_state.c is None

    def test_spawn_new_tree_oldroot(self, tmpdir):
        # TODO: implement
        pass


class TestRecipe:

    def test_bad_recipe(self, tmpdir):
        # This should fail because run does not return a state!
        class Bad(Recipe):

            def run(self, s):
                return None

        exp = Experiment(directory=tmpdir, state_class=EmptyState)
        exp_args = {"a": 10.0, "B": "notused"}
        root = exp.spawn_new_tree(**exp_args)
        with pytest.raises(RuntimeError):
            Bad()(root).get()

    def test_spawn_new_tree_recipe(self, tmpdir):

        exp = Experiment(directory=tmpdir, state_class=EmptyState)
        exp_args = {"a": 10.0, "B": "notused"}
        root = exp.spawn_new_tree(**exp_args)

        # This recipe sets the non-hashed attribute `c`, so we check that we
        # do state.c = 10 * 1.5
        op = OpRecipe(mul, 1.5)

        result = op(root)

        out = result.get()

        # When we `get` the result, it will be slim-loaded, verify that and
        # restore the output.
        assert out.slim_loaded
        assert out.restore()
        assert not out.slim_loaded

        # Very the Op was applied
        assert out.c == 15.0

    def test_cache_works(self, tmpdir):

        exp = Experiment(directory=tmpdir, state_class=EmptyState)

        exp_args = {"a": 10.0, "B": "notused"}
        root = exp.spawn_new_tree(**exp_args)

        # In the stochastic version of this, the output of the Op will change
        # if `run` is called again. To make sure we're using the cached nodes,
        # we check to make sure multiple runs yield the same output.
        op = OpRecipe(mul, 0.4, stochastic=True)
        result = op(root)
        out = result.get()
        assert out.restore()

        value = out.c

        # Get the value a second time by re-running through the graph, assert
        # the same
        result2 = op(root)
        out2 = result2.get()
        assert out2.restore()

        assert value == out2.c

        # Remove all cached states, rebuild the experiment, and assert the
        # value changes.
        for f in glob.glob(str(tmpdir / "*")):
            os.remove(f)

        # Now recreate
        exp = Experiment(directory=tmpdir, state_class=EmptyState)

        exp_args = {"a": 10.0, "B": "notused"}
        root = exp.spawn_new_tree(**exp_args)

        result3 = op(root)
        out3 = result3.get()
        assert out3.restore()

        assert value != out3.c

    def test_same_recipe(self, tmpdir):

        exp = Experiment(directory=tmpdir, state_class=EmptyState)

        exp_args = {"a": 10.0, "B": "notused"}
        root = exp.spawn_new_tree(**exp_args)

        op = OpRecipe(mul, 0.4, stochastic=False)

        new_state_a = op(root)
        new_state_b = op(root)
        new_state_c = op(root)

        # make sure this creates three new dask delayed states
        assert len(exp.leaves) == 3
        for leafID, leaf in exp.leaves.items():
            assert type(leaf) == Delayed

        # Since this is the same op on the root, these three ops would result
        # in 3 identical states. Check, therefore, that only one state is
        # created
        exp.run()
        exp = Experiment.restore(directory=tmpdir, state_class=EmptyState)
        assert len(exp.graph.node_map) == 2  # root + new state

        # Test actual restoring from cache by hand by replicating what
        # happens in `run_recipe`
        exp1 = Experiment(directory=tmpdir, state_class=EmptyState)
        root1 = exp1.spawn_new_tree(**exp_args)
        assert root1.get().from_cache  # same root
        new_state1 = root1.get().new_state(op)
        assert new_state1.restore()


class TestFunction:

    def test_function(self, tmpdir):
        exp = Experiment(directory=tmpdir, state_class=EmptyState)
        exp_args = {"a": 1, "B": 2}
        root = exp.spawn_new_tree(**exp_args)
        op = OpRecipe(mul, 0.4)
        state1 = op(root)

        assert len(exp.leaves) == 1
        leaf1 = list(exp.leaves.items())[0]

        function = Function(lambda s: print("c = {}".format(s.c)))

        state2 = function(state1)

        # The state should not be modified by the function because functions
        # are non state mutating operations
        assert state2 == state1

        # check that the previous leaf has been replaced by the new leaf
        assert len(exp.leaves) == 1
        leaf2 = list(exp.leaves.items())[0]
        assert leaf2 != leaf1

    def test_function_exception(self, tmpdir):
        """Test that when the function fails, the relative error gets raised
        upon running the graph (not at graph definition time).
        """
        exp = Experiment(directory=tmpdir, state_class=EmptyState)
        exp_args = {"a": 1, "B": 2}
        root = exp.spawn_new_tree(**exp_args)

        function = Function(lambda s: print("d = {}".format(s.d)))

        s = function(root)
        with pytest.raises(AttributeError):
            exp.run()

    def test_function_safe_op(self, tmpdir):
        """Regardless of whether the op in the function fails or succeeds,
        the state it acts on gets deflated.
        """
        exp = Experiment(directory=tmpdir, state_class=EmptyState)
        exp_args = {"a": 1, "B": 2}
        root = exp.spawn_new_tree(**exp_args)
        exp.run()
        exp = Experiment.restore(directory=tmpdir, state_class=EmptyState)

        assert exp.root.slim_loaded

        badfunction = Function(lambda s: print("d = {}".format(s.d)))
        with pytest.raises(AttributeError):
            s = badfunction._safe_op(exp.root)
        assert exp.root.slim_loaded

        goodfunction = Function(lambda s: print("a = {}".format(s.a)))
        s = goodfunction._safe_op(exp.root)
        assert exp.root.slim_loaded


class TestStaticExperimentTree:

    def test_tag_filtering(self, tmpdir):
        exp = Experiment(directory=tmpdir, state_class=EmptyState)
        exp_args = {"a": 1.0, "B": 2.0, "c": 3.0}
        root = exp.spawn_new_tree(**exp_args)
        op_add = OpRecipe(add, 1.2)
        with exp.tag("ops"):
            with exp.tag("phase:mul"):
                x1 = OpRecipe(mul, 0.4)(root)
                x2 = OpRecipe(mul, 0.5)(root)
            with exp.tag("phase:add"):
                y1 = op_add(x1)
                y2 = op_add(x2)
        exp.run()
        exp = Experiment.restore(directory=tmpdir, state_class=EmptyState)
        assert len(exp.graph.nodes.filter("op*")) == 4
        assert (
            len(
                exp.graph.nodes.filter("phase:mul")
                | exp.graph.nodes.filter("phase:add")
            )
            == 4
        )
        assert len(exp.graph.nodes.filter("!phase:mul")) == 3
        assert (
            len(
                exp.graph.nodes.filter("ops")
                & exp.graph.nodes.filter("!phase:add")
            )
            == 2
        )

        # Cannot compose other objects with a nodeset
        with pytest.raises(TypeError):
            exp.graph.nodes.filter("phase:mul") | "hi"
        with pytest.raises(TypeError):
            exp.graph.nodes.filter("phase:*") & "!hi"
