from imagiq.common import uid_generator


class Federation:
    def __init__(self, nodes=[], task="", endpoints=[], description="", name=None):
        self.uid = uid_generator()
        self.name = name
        if name is None:
            self.name = "fed_" + self.uid[:5]
        self.nodes = nodes
        self.task = task
        self.endpoints = endpoints
        self.description = description

    def add_node(self, node):
        self.nodes.append(node)

    def __repr__(self):
        return f"Federation {self.uid}"

    def __str__(self):
        retVal = "=" * 79 + "\n"
        retVal += "Federation {}\n".format(self.uid)
        if len(self.nodes) > 0:
            retVal += "\n"
            retVal += "Member Nodes:\n"
            for i, node in enumerate(self.nodes):
                retVal += "\tNode #{:%03d}: {} ({})\n".format(
                    i, node.getName(), node.getUID()
                )
            retVal += "Total {} registered nodes.".format(len(self.nodes))
        return retVal

    def set_description(self, description):
        self.description = description

    def request_to_join(self, node_uid):
        """Approve if node_uid can be accepted to the federation.

        Args:
            node_uid: Public key of the node trying to join.
        Returns:
            Boolean: True if approved.
        """
        # check blacklist
        if node_uid in self.blacklist:
            return False

        # TODO: other criteria?

        # TODO: if passed, send message to all of its member nodes.
        # for node in nodes:
        #     self.send_update(node)
        # TODO: send reply to the node that is applying.
