from collections import deque
from typing import List, Optional


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None) -> None:
        self.val = val
        self.next = next


class TreeNode:
    def __init__(
        self, val: int = 0, left: Optional["TreeNode"] = None, right: Optional["TreeNode"] = None
    ) -> None:
        self.val = val
        self.left = left
        self.right = right


class Solution1:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(-1)
        current = dummy
        while list1 and list2:
            if list1.val <= list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next
            current = current.next
        if list1:
            current.next = list1
        else:
            current.next = list2
        return dummy.next

    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return False
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False


def build_linked_list(values):
    dummy = ListNode(-1)
    current = dummy
    for value in values:
        current.next = ListNode(value)
        current = current.next
    return dummy.next


def linked_list_to_list(node):
    values = []
    while node:
        values.append(node.val)
        node = node.next
    return values


class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        curr = head
        while curr:
            next_temp = curr.next
            curr.next = prev
            prev = curr
            curr = next_temp

        return prev


class Solution3:
    def maxProfit(self, prices: List[int]) -> int:
        min_price = float('inf')
        max_profit = 0
        for price in prices:
            min_price = min(min_price, price)
            max_profit = max(max_profit, price - min_price)

        return max_profit


class Solution4:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root


class Solution5:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        queue = deque([root])
        result = []

        while queue:
            current_level_size = len(queue)
            current_level_nodes = []
            for _ in range(current_level_size):
                node = queue.popleft()
                current_level_nodes.append(node.val)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(current_level_nodes)
        return result


class Solution6:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)
        return 1 + max(left_depth, right_depth)


if __name__ == "__main__":
    # Merge two sorted linked lists
    list1 = build_linked_list([1, 2, 4])
    list2 = build_linked_list([1, 3, 4])
    linked_list_solver = Solution1()
    merged_head = linked_list_solver.mergeTwoLists(list1, list2)
    print("Merged lists:", linked_list_to_list(merged_head))

    # Cycle detection
    cycle_head = build_linked_list([1, 2, 3])
    cycle_tail = cycle_head
    while cycle_tail.next:
        cycle_tail = cycle_tail.next
    cycle_tail.next = cycle_head
    print("Has cycle:", linked_list_solver.hasCycle(cycle_head))

    # Reverse a linked list
    reversed_head = Solution().reverseList(build_linked_list([1, 2, 3, 4]))
    print("Reversed list:", linked_list_to_list(reversed_head))

    # Stock max profit
    stock_solver = Solution3()
    print("Max profit:", stock_solver.maxProfit([7, 1, 5, 3, 6, 4]))

    # Binary tree operations
    tree_root = TreeNode(
        4,
        left=TreeNode(2, TreeNode(1), TreeNode(3)),
        right=TreeNode(7, TreeNode(6), TreeNode(9)),
    )
    inverted = Solution4().invertTree(tree_root)
    print("Level order:", Solution5().levelOrder(inverted))
    print("Max depth:", Solution6().maxDepth(inverted))

