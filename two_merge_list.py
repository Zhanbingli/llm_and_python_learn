from typing import Optional
class ListNode:
    def __init__(self, val:int=0, next:Optional['ListNode'] = None) -> None:
        self.val = val
        self.next = next
class Solution1:
    def mergeTwoLists(self, list1:Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
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

list1 = build_linked_list([1, 2, 4])
list2 = build_linked_list([1, 3, 4])
call = Solution1()
merged_head = call.mergeTwoLists(list1, list2)
print(linked_list_to_list(merged_head))

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

