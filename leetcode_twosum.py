class Solution1:
    def twoSum(self, nums: list[int], target: int) ->list[int]:
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                if nums[i] + nums[j] == target:
                    return [i,j]

        return []


nums = [2, 4, 6, 8, 9]
target = 10
#create an instance first, then call the method
solution = Solution1()
print(solution.twoSum(nums, target))

class Solution2:
    @staticmethod
    def twoSum(nums: list[int], target: int) -> list[int]:
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                if nums[i] + nums[j] == target:
                    return [i, j]

        return []

nums = [1, 3, 5, 7, 9]
target = 10
#can now call directly on the class
print(Solution2.twoSum(nums, target))

from typing import List

class Solution3:
    """feedback first anwser"""
    def twoSum(self, nums:list[int], target: int) ->list[int]:
        num_to_index = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in num_to_index:
                return [num_to_index[complement], i]
            num_to_index[num] = i

        return []

    def twoSumAllPairs(self, nums:List[int], target: int) -> List[List[int]]:
        num_to_index = {}
        pairs = []

        for i, num in enumerate(nums):
            complement = target - num
            if complement in num_to_index:
                pairs.append([num_to_index[complement], i])
            num_to_index[num] = i

        return pairs


nums = [2, 3, 5, 4, 1]
target = 5
solution = Solution3()
print(solution.twoSum(nums, target))
