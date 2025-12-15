//! A DOM-like tree data structure based on `&Node` references.
//!
//! Included from <https://github.com/SimonSapin/rust-forest/blob/5783c8be8680b84c0438638bdee07d4e4aca40ac/arena-tree/lib.rs>.
//! MIT license (per Cargo.toml).
//!
//! Any non-trivial tree involves reference cycles
//! (e.g. if a node has a first child, the parent of the child is that node).
//! To enable this, nodes need to live in an arena allocator
//! such as `arena::TypedArena` distributed with rustc (which is `#[unstable]` as of this writing)
//! or [`typed_arena::Arena`](https://crates.io/crates/typed-arena).
//!
//! If you need mutability in the node’s `data`,
//! make it a cell (`Cell` or `RefCell`) or use cells inside of it.

use std::cell::{BorrowError, Cell, Ref, RefCell, RefMut};
use std::fmt;

/// A node inside a DOM-like tree.
pub struct Node<'a, T: 'a> {
    parent: Cell<Option<&'a Node<'a, T>>>,
    previous_sibling: Cell<Option<&'a Node<'a, T>>>,
    next_sibling: Cell<Option<&'a Node<'a, T>>>,
    first_child: Cell<Option<&'a Node<'a, T>>>,
    last_child: Cell<Option<&'a Node<'a, T>>>,

    /// The data held by the node.
    pub data: T,
}

#[cfg(feature = "serde")]
mod serde_impls {
    use super::Node;
    use std::collections::HashMap;

    use serde::{de::Error as DeError, Deserialize, Serialize, Serializer};
    use typed_arena::Arena;

    /// Serializable representation of a Node for serde.
    ///
    /// Avoids cycles during serialization/deserialization by using integer
    /// indices instead of actual references. These are resolved during
    /// deserialization into actual references after all nodes are created.
    ///
    /// All indices are relative to the [`WireTree::nodes`] array, which holds
    /// all nodes in the tree and acts as a temporary arena for this purpose.
    #[derive(Default, Serialize, Deserialize)]
    struct WireNode<T> {
        id: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        previous_sibling: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        next_sibling: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        first_child: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        last_child: Option<u32>,
        data: T,
    }

    /// Serializable representation of an arena tree for serde.
    ///
    /// Holds all nodes in a flat vector to avoid cycles during
    /// serialization and deserialization. References between nodes are represented
    /// as integer indices in the array. These are resolved during the final
    /// resolution step of deserialization, after all nodes are created, into
    /// actual references and [`Node`] structs. This allows serde to serialize
    /// and deserialize the [`Node`] tree structure without running into issues
    /// with reference cycles.
    #[derive(Default, Serialize, Deserialize)]
    struct WireTree<T> {
        nodes: Vec<WireNode<T>>,
    }

    impl<'a, T: 'a> Serialize for Node<'a, T>
    where
        T: Serialize,
    {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            // Force the borrow of `self` to match the lifetime carried by the
            // node so it can be threaded through the traversal.
            let root: &'a Node<'a, T> = unsafe { std::mem::transmute::<_, &'a Node<'a, T>>(self) };
            let mut ordered = Vec::new();
            for node in root.descendants() {
                ordered.push(node);
            }

            let mut ids = HashMap::with_capacity(ordered.len());
            for (idx, node) in ordered.iter().enumerate() {
                ids.insert(std::ptr::from_ref(*node) as usize, idx as u32);
            }

            let index_for = |node: Option<&'a Node<'a, T>>| {
                node.map(|n| ids[&(std::ptr::from_ref(n) as usize)])
            };

            WireTree {
                nodes: ordered
                    .iter()
                    .map(|node| WireNode {
                        id: index_for(Some(node)).unwrap(),
                        parent: index_for(node.parent()),
                        previous_sibling: index_for(node.previous_sibling()),
                        next_sibling: index_for(node.next_sibling()),
                        first_child: index_for(node.first_child()),
                        last_child: index_for(node.last_child()),
                        data: &node.data,
                    })
                    .collect::<Vec<_>>(),
            }
            .serialize(serializer)
        }
    }

    impl<'a, 'de, T: 'a + 'de> serde::Deserialize<'de> for &'a Node<'a, T>
    where
        T: serde::Deserialize<'de>,
    {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let WireTree { nodes: wire_nodes } = WireTree::<T>::deserialize(deserializer)?;
            if wire_nodes.is_empty() {
                return Err(DeError::custom("cannot deserialize empty tree"));
            }

            let arena = Box::leak(Box::new(Arena::<Node<'de, T>>::new()));
            let mut nodes: Vec<&Node<'de, T>> = Vec::with_capacity(wire_nodes.len());
            let mut peers = Vec::with_capacity(wire_nodes.len());

            for wire in wire_nodes.into_iter() {
                peers.push((
                    wire.parent,
                    wire.previous_sibling,
                    wire.next_sibling,
                    wire.first_child,
                    wire.last_child,
                ));
                nodes.push(arena.alloc(Node::new(wire.data)));
            }

            for (node, peer) in nodes.iter().zip(peers.into_iter()) {
                let (parent, prev, next, first, last) = peer;
                let parent_ref =
                    match parent {
                        Some(i) => Some(*nodes.get(i as usize).ok_or_else(|| {
                            DeError::custom(format!("node index {i} out of bounds"))
                        })?),
                        None => None,
                    };
                let prev_ref =
                    match prev {
                        Some(i) => Some(*nodes.get(i as usize).ok_or_else(|| {
                            DeError::custom(format!("node index {i} out of bounds"))
                        })?),
                        None => None,
                    };
                let next_ref =
                    match next {
                        Some(i) => Some(*nodes.get(i as usize).ok_or_else(|| {
                            DeError::custom(format!("node index {i} out of bounds"))
                        })?),
                        None => None,
                    };
                let first_ref =
                    match first {
                        Some(i) => Some(*nodes.get(i as usize).ok_or_else(|| {
                            DeError::custom(format!("node index {i} out of bounds"))
                        })?),
                        None => None,
                    };
                let last_ref =
                    match last {
                        Some(i) => Some(*nodes.get(i as usize).ok_or_else(|| {
                            DeError::custom(format!("node index {i} out of bounds"))
                        })?),
                        None => None,
                    };

                node.parent.set(parent_ref);
                node.previous_sibling.set(prev_ref);
                node.next_sibling.set(next_ref);
                node.first_child.set(first_ref);
                node.last_child.set(last_ref);
            }

            let root = nodes[0];
            // SAFETY: The arena is leaked, so the references live for the
            // remainder of the program. Since `T: 'a + 'de`, the data inside
            // the nodes also live for 'a. Thus, the whole tree rooted at
            // `root` lives for 'a; shortening the lifetime should be safe.
            Ok(unsafe { core::mem::transmute::<_, &Node<'a, T>>(root) })
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::nodes::Ast;

        use super::super::*;
        use serde_json;
        #[test]
        fn basic_serialization_roundtrip() {
            let arena = typed_arena::Arena::new();
            let root = arena.alloc(Node::new("root"));
            let child1 = arena.alloc(Node::new("child1"));
            let child2 = arena.alloc(Node::new("child2"));
            root.append(child1);
            root.append(child2);

            let serialized = serde_json::to_string(&root).unwrap();
            let deserialized: &Node<'_, &str> = serde_json::from_str(&serialized).unwrap();

            assert_eq!(deserialized.data, "root");

            let mut children = deserialized.children();
            assert_eq!(children.next().unwrap().data, "child1");
            assert_eq!(children.next().unwrap().data, "child2");
            assert!(children.next().is_none());
        }

        #[test]
        fn empty_tree_serialization_fails() {
            let result: Result<&Node<'_, &str>, _> = serde_json::from_str(r#"{"nodes":[]}"#);
            assert!(result.is_err());
        }

        #[test]
        fn out_of_bounds_index_fails() {
            let result: Result<&Node<'_, &str>, _> = serde_json::from_str(
                r#"{
    "nodes": [
        {
            "id": 0,
            "first_child": 1,
            "last_child": 1,
            "data": "root"
        }
    ]
}"#,
            );
            assert!(result.is_err());
        }

        #[test]
        /// A more complex tree to ensure references are correctly resolved
        /// when deserializing a tree with multiple levels and siblings, much
        /// like our actual parsed AST structures.
        fn complex_tree_serialization_roundtrip() {
            use crate::nodes::NodeValue;
            use crate::options::*;
            use crate::parse_document;
            use crate::Arena;
            let arena = Arena::new();
            let root = parse_document(
                &arena,
                r#"# Hello World

## This is a subtitle

And this a paragraph.
"#,
                &Options::default(),
            );

            let serialized = serde_json::to_string_pretty(&root).unwrap();
            let deserialized: &Node<'_, RefCell<Ast>> = serde_json::from_str(&serialized).unwrap();

            // Check root node
            assert!(matches!(deserialized.data().value, NodeValue::Document));

            // Check first child (heading 1)
            let mut children = deserialized.children();

            let heading1 = children.next().unwrap();
            assert!(matches!(heading1.data().value, NodeValue::Heading(_)));

            let heading1_text = heading1.first_child().unwrap();
            assert!(
                matches!(heading1_text.data().value, NodeValue::Text(ref text) if *text == "Hello World")
            );

            // Check second child (heading 2)
            let heading2 = children.next().unwrap();
            assert!(matches!(heading2.data().value, NodeValue::Heading(_)));

            let heading2_text = heading2.first_child().unwrap();
            assert!(
                matches!(heading2_text.data().value, NodeValue::Text(ref text) if *text == "This is a subtitle")
            );

            // Check third child (paragraph)
            let paragraph = children.next().unwrap();
            assert!(matches!(paragraph.data().value, NodeValue::Paragraph));

            let paragraph_text = paragraph.first_child().unwrap();
            assert!(
                matches!(paragraph_text.data().value, NodeValue::Text(ref text) if *text == "And this a paragraph.")
            );

            assert!(children.next().is_none());
        }
    }
}

/// A simple Debug implementation that prints the children as a tree, without
/// looping through the various interior pointer cycles.
impl<'a, T: 'a> fmt::Debug for Node<'a, RefCell<T>>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        struct Children<'a, T>(Option<&'a Node<'a, RefCell<T>>>);
        impl<T: fmt::Debug> fmt::Debug for Children<'_, T> {
            fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
                f.debug_list()
                    .entries(std::iter::successors(self.0, |child| {
                        child.next_sibling.get()
                    }))
                    .finish()
            }
        }

        if let Ok(data) = self.data.try_borrow() {
            write!(f, "{data:?}")?;
        } else {
            write!(f, "!!mutably borrowed!!")?;
        }
        if let Some(first_child) = self.first_child.get() {
            write!(f, " {:?}", &Children(Some(first_child)))?;
        }
        Ok(())
    }
}

impl<'a, T> Node<'a, T> {
    /// Create a new node from its associated data.
    ///
    /// Typically, this node needs to be moved into an arena allocator
    /// before it can be used in a tree.
    pub const fn new(data: T) -> Node<'a, T> {
        Node {
            parent: Cell::new(None),
            first_child: Cell::new(None),
            last_child: Cell::new(None),
            previous_sibling: Cell::new(None),
            next_sibling: Cell::new(None),
            data,
        }
    }

    /// Return a reference to the parent node, unless this node is the root of the tree.
    pub const fn parent(&self) -> Option<&'a Node<'a, T>> {
        self.parent.get()
    }

    /// Return a reference to the first child of this node, unless it has no child.
    pub const fn first_child(&self) -> Option<&'a Node<'a, T>> {
        self.first_child.get()
    }

    /// Return a reference to the last child of this node, unless it has no child.
    pub const fn last_child(&self) -> Option<&'a Node<'a, T>> {
        self.last_child.get()
    }

    /// Return a reference to the previous sibling of this node, unless it is a first child.
    pub const fn previous_sibling(&self) -> Option<&'a Node<'a, T>> {
        self.previous_sibling.get()
    }

    /// Return a reference to the next sibling of this node, unless it is a last child.
    pub const fn next_sibling(&self) -> Option<&'a Node<'a, T>> {
        self.next_sibling.get()
    }

    /// Returns whether two references point to the same node.
    pub fn same_node(&self, other: &Node<'a, T>) -> bool {
        std::ptr::eq(self, other)
    }

    /// Return an iterator of references to this node and its ancestors.
    ///
    /// Call `.next().unwrap()` once on the iterator to skip the node itself.
    pub const fn ancestors(&'a self) -> Ancestors<'a, T> {
        Ancestors(Some(self))
    }

    /// Return an iterator of references to this node and the siblings before it.
    ///
    /// Call `.next().unwrap()` once on the iterator to skip the node itself.
    pub const fn preceding_siblings(&'a self) -> PrecedingSiblings<'a, T> {
        PrecedingSiblings(Some(self))
    }

    /// Return an iterator of references to this node and the siblings after it.
    ///
    /// Call `.next().unwrap()` once on the iterator to skip the node itself.
    pub const fn following_siblings(&'a self) -> FollowingSiblings<'a, T> {
        FollowingSiblings(Some(self))
    }

    /// Return an iterator of references to this node’s children.
    pub const fn children(&'a self) -> Children<'a, T> {
        Children(self.first_child.get())
    }

    /// Return an iterator of references to this node’s children, in reverse order.
    pub const fn reverse_children(&'a self) -> ReverseChildren<'a, T> {
        ReverseChildren(self.last_child.get())
    }

    /// Return an iterator of references to this `Node` and its descendants, in tree order.
    ///
    /// Parent nodes appear before the descendants.
    /// Call `.next().unwrap()` once on the iterator to skip the node itself.
    ///
    /// *Similar Functions:* Use `traverse()` or `reverse_traverse` if you need
    /// references to the `NodeEdge` structs associated with each `Node`
    pub const fn descendants(&'a self) -> Descendants<'a, T> {
        Descendants(self.traverse())
    }

    /// Return an iterator of references to `NodeEdge` enums for each `Node` and its descendants,
    /// in tree order.
    ///
    /// `NodeEdge` enums represent the `Start` or `End` of each node.
    ///
    /// *Similar Functions:* Use `descendants()` if you don't need `Start` and `End`.
    pub const fn traverse(&'a self) -> Traverse<'a, T> {
        Traverse {
            root: self,
            next: Some(NodeEdge::Start(self)),
        }
    }

    /// Return an iterator of references to `NodeEdge` enums for each `Node` and its descendants,
    /// in *reverse* order.
    ///
    /// `NodeEdge` enums represent the `Start` or `End` of each node.
    ///
    /// *Similar Functions:* Use `descendants()` if you don't need `Start` and `End`.
    pub const fn reverse_traverse(&'a self) -> ReverseTraverse<'a, T> {
        ReverseTraverse {
            root: self,
            next: Some(NodeEdge::End(self)),
        }
    }

    /// Detach a node from its parent and siblings. Children are not affected.
    pub fn detach(&self) {
        let parent = self.parent.take();
        let previous_sibling = self.previous_sibling.take();
        let next_sibling = self.next_sibling.take();

        if let Some(next_sibling) = next_sibling {
            next_sibling.previous_sibling.set(previous_sibling);
        } else if let Some(parent) = parent {
            parent.last_child.set(previous_sibling);
        }

        if let Some(previous_sibling) = previous_sibling {
            previous_sibling.next_sibling.set(next_sibling);
        } else if let Some(parent) = parent {
            parent.first_child.set(next_sibling);
        }
    }

    /// Append a new child to this node, after existing children.
    pub fn append(&'a self, new_child: &'a Node<'a, T>) {
        new_child.detach();
        new_child.parent.set(Some(self));
        if let Some(last_child) = self.last_child.take() {
            new_child.previous_sibling.set(Some(last_child));
            debug_assert!(last_child.next_sibling.get().is_none());
            last_child.next_sibling.set(Some(new_child));
        } else {
            debug_assert!(self.first_child.get().is_none());
            self.first_child.set(Some(new_child));
        }
        self.last_child.set(Some(new_child));
    }

    /// Append multiple new children to this node, after existing children.
    pub fn extend(&'a self, new_children: impl IntoIterator<Item = &'a Node<'a, T>>) {
        for child in new_children.into_iter() {
            self.append(child);
        }
    }

    /// Prepend a new child to this node, before existing children.
    pub fn prepend(&'a self, new_child: &'a Node<'a, T>) {
        new_child.detach();
        new_child.parent.set(Some(self));
        if let Some(first_child) = self.first_child.take() {
            debug_assert!(first_child.previous_sibling.get().is_none());
            first_child.previous_sibling.set(Some(new_child));
            new_child.next_sibling.set(Some(first_child));
        } else {
            debug_assert!(self.first_child.get().is_none());
            self.last_child.set(Some(new_child));
        }
        self.first_child.set(Some(new_child));
    }

    /// Insert a new sibling after this node.
    pub fn insert_after(&'a self, new_sibling: &'a Node<'a, T>) {
        new_sibling.detach();
        new_sibling.parent.set(self.parent.get());
        new_sibling.previous_sibling.set(Some(self));
        if let Some(next_sibling) = self.next_sibling.take() {
            debug_assert!(std::ptr::eq(
                next_sibling.previous_sibling.get().unwrap(),
                self
            ));
            next_sibling.previous_sibling.set(Some(new_sibling));
            new_sibling.next_sibling.set(Some(next_sibling));
        } else if let Some(parent) = self.parent.get() {
            debug_assert!(std::ptr::eq(parent.last_child.get().unwrap(), self));
            parent.last_child.set(Some(new_sibling));
        }
        self.next_sibling.set(Some(new_sibling));
    }

    /// Insert a new sibling before this node.
    pub fn insert_before(&'a self, new_sibling: &'a Node<'a, T>) {
        new_sibling.detach();
        new_sibling.parent.set(self.parent.get());
        new_sibling.next_sibling.set(Some(self));
        if let Some(previous_sibling) = self.previous_sibling.take() {
            new_sibling.previous_sibling.set(Some(previous_sibling));
            debug_assert!(std::ptr::eq(
                previous_sibling.next_sibling.get().unwrap(),
                self
            ));
            previous_sibling.next_sibling.set(Some(new_sibling));
        } else if let Some(parent) = self.parent.get() {
            debug_assert!(std::ptr::eq(parent.first_child.get().unwrap(), self));
            parent.first_child.set(Some(new_sibling));
        }
        self.previous_sibling.set(Some(new_sibling));
    }
}

macro_rules! axis_iterator {
    (#[$attr:meta] $name:ident : $next:ident) => {
        #[$attr]
        pub struct $name<'a, T: 'a>(Option<&'a Node<'a, T>>);

        impl<'a, T: 'a> fmt::Debug for $name<'a, RefCell<T>>
        where
            T: fmt::Debug,
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_tuple(stringify!($name)).field(&self.0).finish()
            }
        }

        impl<'a, T> Iterator for $name<'a, T> {
            type Item = &'a Node<'a, T>;

            fn next(&mut self) -> Option<&'a Node<'a, T>> {
                match self.0.take() {
                    Some(node) => {
                        self.0 = node.$next.get();
                        Some(node)
                    }
                    None => None,
                }
            }
        }
    };
}

axis_iterator! {
    #[doc = "An iterator of references to the ancestors a given node."]
    Ancestors: parent
}

axis_iterator! {
    #[doc = "An iterator of references to the siblings before a given node."]
    PrecedingSiblings: previous_sibling
}

axis_iterator! {
    #[doc = "An iterator of references to the siblings after a given node."]
    FollowingSiblings: next_sibling
}

axis_iterator! {
    #[doc = "An iterator of references to the children of a given node."]
    Children: next_sibling
}

axis_iterator! {
    #[doc = "An iterator of references to the children of a given node, in reverse order."]
    ReverseChildren: previous_sibling
}

/// An iterator of references to a given node and its descendants, in tree order.
pub struct Descendants<'a, T: 'a>(Traverse<'a, T>);

impl<'a, T: 'a> fmt::Debug for Descendants<'a, RefCell<T>>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Descendants").field(&self.0).finish()
    }
}

impl<'a, T> Iterator for Descendants<'a, T> {
    type Item = &'a Node<'a, T>;

    fn next(&mut self) -> Option<&'a Node<'a, T>> {
        loop {
            match self.0.next() {
                Some(NodeEdge::Start(node)) => return Some(node),
                Some(NodeEdge::End(_)) => {}
                None => return None,
            }
        }
    }
}

/// An edge of the node graph returned by a traversal iterator.
#[derive(Debug, Clone)]
pub enum NodeEdge<T> {
    /// Indicates that start of a node that has children.
    /// Yielded by `Traverse::next` before the node’s descendants.
    /// In HTML or XML, this corresponds to an opening tag like `<div>`
    Start(T),

    /// Indicates that end of a node that has children.
    /// Yielded by `Traverse::next` after the node’s descendants.
    /// In HTML or XML, this corresponds to a closing tag like `</div>`
    End(T),
}

macro_rules! traverse_iterator {
    (#[$attr:meta] $name:ident : $first_child:ident, $next_sibling:ident) => {
        #[$attr]
        pub struct $name<'a, T: 'a> {
            root: &'a Node<'a, T>,
            next: Option<NodeEdge<&'a Node<'a, T>>>,
        }

        impl<'a, T: 'a> fmt::Debug for $name<'a, RefCell<T>>
        where
            T: fmt::Debug,
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($name))
                    .field("root", &self.root)
                    .field("next", &self.next)
                    .finish()
            }
        }

        impl<'a, T> Iterator for $name<'a, T> {
            type Item = NodeEdge<&'a Node<'a, T>>;

            fn next(&mut self) -> Option<NodeEdge<&'a Node<'a, T>>> {
                match self.next.take() {
                    Some(item) => {
                        self.next = match item {
                            NodeEdge::Start(node) => match node.$first_child.get() {
                                Some(child) => Some(NodeEdge::Start(child)),
                                None => Some(NodeEdge::End(node)),
                            },
                            NodeEdge::End(node) => {
                                if node.same_node(self.root) {
                                    None
                                } else {
                                    match node.$next_sibling.get() {
                                        Some(sibling) => Some(NodeEdge::Start(sibling)),
                                        None => match node.parent.get() {
                                            Some(parent) => Some(NodeEdge::End(parent)),
                                            None => panic!("tree modified during iteration"),
                                        },
                                    }
                                }
                            }
                        };
                        Some(item)
                    }
                    None => None,
                }
            }
        }
    };
}

traverse_iterator! {
    #[doc = "An iterator of the start and end edges of a given
    node and its descendants, in tree order."]
    Traverse: first_child, next_sibling
}

traverse_iterator! {
    #[doc = "An iterator of the start and end edges of a given
    node and its descendants, in reverse tree order."]
    ReverseTraverse: last_child, previous_sibling
}

impl<'a, T> Node<'a, RefCell<T>> {
    /// Shorthand for `node.data.borrow()`.
    pub fn data(&self) -> Ref<'_, T> {
        self.data.borrow()
    }

    /// Shorthand for `node.data.try_borrow()`.
    pub fn try_data(&self) -> Result<Ref<'_, T>, BorrowError> {
        self.data.try_borrow()
    }

    /// Shorthand for `node.data.borrow_mut()`.
    pub fn data_mut(&self) -> RefMut<'_, T> {
        self.data.borrow_mut()
    }
}

#[test]
fn it_works() {
    struct DropTracker<'a>(&'a Cell<u32>);
    impl<'a> Drop for DropTracker<'a> {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    let drop_counter = Cell::new(0);
    {
        let mut new_counter = 0;
        let arena = typed_arena::Arena::new();
        let mut new = || {
            new_counter += 1;
            arena.alloc(Node::new((new_counter, DropTracker(&drop_counter))))
        };

        let a = new(); // 1
        a.append(new()); // 2
        a.append(new()); // 3
        a.prepend(new()); // 4
        let b = new(); // 5
        b.append(a);
        a.insert_before(new()); // 6
        a.insert_before(new()); // 7
        a.insert_after(new()); // 8
        a.insert_after(new()); // 9
        let c = new(); // 10
        b.append(c);

        assert_eq!(drop_counter.get(), 0);
        c.previous_sibling.get().unwrap().detach();
        assert_eq!(drop_counter.get(), 0);

        assert_eq!(
            b.descendants().map(|node| node.data.0).collect::<Vec<_>>(),
            [5, 6, 7, 1, 4, 2, 3, 9, 10]
        );
    }

    assert_eq!(drop_counter.get(), 10);
}
