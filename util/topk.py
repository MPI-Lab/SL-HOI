from typing import List

def sift_down(heap: List[int], start: int, end: int) -> None:
    """
    Maintains the min-heap property for a subtree rooted at index `start`.

    This function is used in the Top-K selection algorithm to ensure the min-heap property
    after updating the root element.

    Args:
        heap (List[int]): The heap represented as a list.
        start (int): The index of the root of the subtree.
        end (int): The last index of the heap.

    References:
        - Heap sort and Top-K selection via min-heap.
        - Maintains: heap[start] <= heap[child] for all children.
    """
    temp = heap[start]
    i = start
    j = 2 * i + 1  # Left child

    while j <= end:
        # Select the smaller child
        if j + 1 <= end and heap[j + 1] < heap[j]:
            j += 1
        if temp > heap[j]:
            heap[i] = heap[j]
            i = j
            j = 2 * i + 1
        else:
            break  # Found correct position for temp
    heap[i] = temp

def top_k(li: List[int], k: int) -> List[int]:
    """
    Finds the Top-K largest elements in `li` using a min-heap of size K.

    This is equivalent to the "heap-based selection" algorithm, which maintains a min-heap
    of the K largest seen elements. At the end, the heap contains the K largest elements
    in ascending order.

    Args:
        li (List[int]): Input list of numbers. Length: N.
        k (int): The number of top elements to select.

    Returns:
        List[int]: The Top-K largest elements in ascending order.

    References:
        - This is the classical min-heap Top-K selection algorithm.
        - Theoretical runtime: O(N log K)
        - See CLRS: "Heap Data Structures" and "Selection Algorithms".

    Example:
        >>> top_k([3, 1, 5, 2, 9, 8, 7], 3)
        [7, 8, 9]
    """
    if k <= 0 or k > len(li):
        raise ValueError("k must be in the range 1 <= k <= len(li)")

    # Step 1: Build min-heap from first k elements
    heap = li[:k]
    for i in range(k // 2 - 1, -1, -1):
        sift_down(heap, i, k - 1)

    # Step 2: Iterate through remaining elements and maintain heap
    for i in range(k, len(li)):
        if li[i] > heap[0]:
            heap[0] = li[i]
            sift_down(heap, 0, k - 1)

    # Step 3: Sort the heap in ascending order (in-place heap sort)
    for i in range(k - 1, 0, -1):
        heap[0], heap[i] = heap[i], heap[0]
        sift_down(heap, 0, i - 1)

    # After heap sort, heap is sorted in ascending order
    return heap
