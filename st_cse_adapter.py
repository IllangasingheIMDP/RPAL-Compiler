"""
ST to CSE Adapter Module

This module provides an adapter to convert the Standardized Tree (ST) nodes from the standardizer
to a format that the CSE machine's linearizer can understand.
"""

class STNodeAdapter:
    """
    Adapter class to convert STNode objects to a format compatible with the CSE machine's linearizer.
    
    The standardizer uses STNode objects with 'node_type' and 'value' attributes,
    but the linearizer expects objects with 'data' attribute.
    """
    
    def __init__(self, st_node):
        """
        Initialize the adapter with an STNode from the standardizer.
        
        Args:
            st_node: An STNode object from the standardizer.
        """
        self.original_node = st_node
        
        # Map node_type to data
        if st_node.node_type == "identifier":
            self.data = "<ID:" + str(st_node.value) + ">"
        elif st_node.node_type == "integer":
            self.data = "<INT:" + str(st_node.value) + ">"
        elif st_node.node_type == "string":
            self.data = "<STR:'" + str(st_node.value) + "'>"
        elif st_node.node_type == "boolean" and st_node.value == True:
            self.data = "<true>"
        elif st_node.node_type == "boolean" and st_node.value == False:
            self.data = "<false>"
        elif st_node.node_type == "nil":
            self.data = "<nil>"
        else:
            self.data = st_node.node_type
        
        self.value = st_node.value
        self.children = [STNodeAdapter(child) if hasattr(child, 'node_type') else child 
                         for child in st_node.children]
    
    def __str__(self):
        """String representation of the adapter"""
        if self.value is not None:
            return f"{self.data}({self.value})"
        return self.data

def adapt_st_for_cse(st_node):
    """
    Convert an STNode from the standardizer to a format compatible with the CSE machine.
    
    Args:
        st_node: An STNode object from the standardizer.
        
    Returns:
        An STNodeAdapter object that can be used with the CSE machine's linearizer.
    """
    return STNodeAdapter(st_node)

