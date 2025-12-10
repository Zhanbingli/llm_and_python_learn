class Solution1:
    def removeElement(self, nums: List[int], val: int) -> int:
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1

        return slow

    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0

        low = 0
        for fast in range(1,len(nums)):
            if nums[fast] != nums[low]:
                low += 1
                nums[low] = nums[fast]
        return low + 1


class Solution:
    def isVaild(self, s: str) -> bool:
        stack = []
        mapping = {
            ')': '(',
            ']': '[',
            '}': '{'
        }
        for char in s:
            if char in mapping:
                top_element = stack.pop() if not stack else "#"
                if mapping[char] != top_element:
                    return False
            else:
                stack.append(char)
        return not stack

    def isValid(self, s: str) -> bool:
        stack = []
        mapping ={
            '(':')',
            '[':']',
            '{':'}'
        }
        for char in s:
            if char in mapping:
                stack.append(char)
            else:
                if not stack:
                    return False
                top_element = stack.pop()
                if mapping[top_element] != char:
                    return False
        return not stack
