# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import fnmatch
from collections import namedtuple


class StaticExperimentTree(
    namedtuple("StaticExperimentTree", ["node_map", "edge_map", "toposort"])
):
    """
    StaticExperimentTree: Keeps track of objects needed for static graph
    analysis of experiments. Conceptually related to a `dagger` `Experiment`,
    but *not* used by dask to run the computation. A `StaticExperimentTree` is
    simply used for visualization, bookkeeping, and analysis purposes.

    node_map: dict that maps each node sha to the correct experiment state
    edge_map: dict that maps each node sha to its set of children's shas
    toposort: dict of dicts that maps each level in the tree (i.e. distance
        from root) to the set of node shas at that level
    """

    class NodeSet(list):
        """List of nodes in an experiment, augmented with functionalities for
        easy filtering and inspection for analysis purposes.
        """

        def filter(self, pattern):
            """Return all experiment states that match the pattern in their
            tag.
            Args:
                pattern (string): substring to look for regular expression
                    matching
            Returns:
                NodeSet with the states whose tags match the pattern
            """
            negation = pattern.startswith("!")
            if negation:
                pattern = pattern[1:]

            return self.__class__(
                node
                for node in self
                if negation
                ^ any(fnmatch.fnmatch(tag, pattern) for tag in node.tags)
            )

        def __or__(self, nodeset):
            if not isinstance(nodeset, self.__class__):
                raise TypeError(
                    f"Cannot compose with object of type:{type(nodeset)}"
                )

            return self.__class__(
                {node for ns in [self, nodeset] for node in ns}
            )

        def __and__(self, nodeset):
            if not isinstance(nodeset, self.__class__):
                raise TypeError(
                    f"Cannot compose with object of type:{type(nodeset)}"
                )

            return self.__class__(set(self).intersection(nodeset))

        @property
        def iterator(self):
            """Iterator that takes care of restoring and deflating states as
            they get accessed to reduce memory footprint to a minimum while
            looping through states.
            """
            for node in self:
                with node.lazy_load():
                    yield node

    @property
    def nodes(self):
        return self.__class__.NodeSet(self.node_map.values())

    @property
    def root(self):
        topo_zero = list(self.toposort[0])
        if len(topo_zero) != 1:
            raise RuntimeError(
                "Invalid graph - found more than one 'root' per toposort"
            )
        return self.node_map[topo_zero[0]]

    def node(self, sha):
        """Access an experiment state by specifying its hash value.
        Args:
            sha (str): hash that identifies the state that we want to access
        Returns:
            ExperimentState corresponding to specified sha value.
        """
        return self.node_map[sha]

    def nodes_at_distance(self, distance):
        """Access all experiment states in an experiment that are at a
        specific distance from the root state in the experimental tree.
        Args:
            distance (int): depth in the tree from the root to slice the tree
                at.
        Returns:
            NodeSet with all states at that distance from the root in the
                experimental tree.
        """
        if distance not in self.toposort:
            return self.__class__.NodeSet([])
        return self.__class__.NodeSet(
            {self.node_map[sha] for sha in self.toposort[distance]}
        )

    def to_graphviz(self, node_args=None):
        """Constructs a graphviz visual graph of the experiment tree for easy
        visual inspection. Each state is connected by directed arrows to its
        children that were created by acting with a Recipe on the state
        itself. The default appearance will display each node's hash value,
        level, and tags. The appearance of each node can be modified by
        passing in `node_args`.
        Returns:
            A graphviz.Digraph of the StaticExperimentTree
        """
        from graphviz import Digraph

        dot = Digraph(comment="Experiment Graph")

        node_args = node_args or {}

        for distance, levelset in self.toposort.items():
            distance = f"level=<FONT COLOR='magenta'>{distance}</FONT>"
            for sha in levelset:
                node = self.node(sha)
                tags = ""
                if node.tags:
                    tags = [
                        f"<FONT COLOR='red'>'{tag}'</FONT>"
                        for tag in node.tags
                    ]
                    tags = f", tags={', '.join(tags)}"

                dot.node(
                    sha,
                    (
                        f"<<FONT COLOR='blue'>{sha}</FONT> "
                        f"({distance}{tags})>"
                    ),
                    shape="rect",
                    fontname="menlo",
                    **node_args,
                )

        for parent, children in self.edge_map.items():
            for child in children:
                dot.edge(parent, child)

        return dot

    def draw(self, filename="graph", format="pdf", view=False):
        """Draw the graph of the StaticExperimentTree and save it to disk at
        the specified location. If possible and `view` is set to True,
        display it.
        """
        d = self.to_graphviz()

        def _is_notebook():
            """Identify whether the graph should be visualized in a jupyter
            notebook.
            """
            import os

            env = os.environ
            shell = "shell"
            program = os.path.basename(env["_"])

            if "jupyter" in program:
                return True
            if "JPY_PARENT_PID" in env:
                return True

            return False

        if view and _is_notebook():
            d.render(
                filename=filename, format=format, view=False, cleanup=True
            )
            from IPython.display import Image

            return Image(filename + "." + format)
        else:
            return d.render(
                filename=filename, format=format, view=view, cleanup=True
            )
