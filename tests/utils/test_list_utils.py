from morseg.utils.list_utils import (list_contains_sublist, get_start_idx_for_sublist,
                                     remove_repeating_symbols, remove_and_insert_placeholder,
                                     insert_and_remove_placeholder)

import pytest


def test_list_contains_sublist():
    # main list
    main_list = ["a", "b", "c", "d", "e"]

    # CHECKS FOR VALID SUBLISTS (method should return True)
    # check for valid sublist in the middle of the main list
    assert list_contains_sublist(main_list, ["b", "c"])

    # check for valid sublist at the beginning of the main list
    assert list_contains_sublist(main_list, ["a", "b", "c"])

    # check for valid sublist at the end of the main list
    assert list_contains_sublist(main_list, ["d", "e"])

    # check for lists with sizes 0 or 1 (should still return True)
    assert list_contains_sublist(main_list, [])
    assert list_contains_sublist(main_list, ["a"])
    assert list_contains_sublist(main_list, ["d"])
    assert list_contains_sublist(main_list, ["e"])

    # CHECKS FOR INVALID SUBLISTS (methods should return False)
    assert not list_contains_sublist(main_list, ["d", "f"])
    assert not list_contains_sublist(main_list, ["D", "e"])
    assert not list_contains_sublist(main_list, ["a", "b", "c", "d", "e", "f"])
    assert not list_contains_sublist(main_list, [1])
    assert not list_contains_sublist(main_list, "abc")  # should NOT compare string to lists
    assert not list_contains_sublist(main_list, None)


def test_get_start_idx_for_sublist():
    # main list
    main_list = ["a", "b", "c", "d", "e"]

    # CHECKS FOR VALID SUBLISTS (method should return the correct start index)
    # check for valid sublist in the middle of the main list
    assert get_start_idx_for_sublist(main_list, ["b", "c"]) == 1

    # check for valid sublist at the beginning of the main list
    assert get_start_idx_for_sublist(main_list, ["a", "b", "c"]) == 0

    # check for valid sublist at the end of the main list
    assert get_start_idx_for_sublist(main_list, ["d", "e"]) == 3

    # check for lists with sizes 0 or 1 (should still return True)
    assert get_start_idx_for_sublist(main_list, []) == 0
    assert get_start_idx_for_sublist(main_list, ["a"]) == 0
    assert get_start_idx_for_sublist(main_list, ["d"]) == 3
    assert get_start_idx_for_sublist(main_list, ["e"]) == 4

    # CHECKS FOR INVALID SUBLISTS (methods should return -1)
    assert get_start_idx_for_sublist(main_list, ["d", "f"]) == -1
    assert get_start_idx_for_sublist(main_list, ["D", "e"]) == -1
    assert get_start_idx_for_sublist(main_list, ["a", "b", "c", "d", "e", "f"]) == -1
    assert get_start_idx_for_sublist(main_list, [1]) == -1
    assert get_start_idx_for_sublist(main_list, "abc") == -1  # should NOT compare string to lists
    assert get_start_idx_for_sublist(main_list, None) == -1


def test_remove_repeating_symbols():
    ref_list = ["a", "b", "+", "c", "+"]

    # remove repeating symbols at the end of a list
    input_list = ["a", "b", "+", "c", "+", "+", "+", "+"]
    assert remove_repeating_symbols(input_list, "+") == ref_list

    # remove repeating symbols in the middle of a list
    input_list = ["a", "b", "+", "+", "c", "+", "+", "+", "+"]
    assert remove_repeating_symbols(input_list, "+") == ref_list

    # remove repeating symbols at the beginning of a list
    ref_list = ["+", "a", "b", "+", "c"]
    input_list = ["+", "+", "+", "+", "a", "b", "+", "+", "c"]
    assert remove_repeating_symbols(input_list, "+") == ref_list

    # target symbol not in list
    assert remove_repeating_symbols(input_list, "-") == input_list

    # no repetitions of target symbol
    assert remove_repeating_symbols(ref_list, "+") == ref_list

    # handling None input
    assert remove_repeating_symbols(None, "+") is None
    assert remove_repeating_symbols(ref_list, None) == ref_list


def test_insert_and_remove_placeholder():
    list_with_placeholder = ["a", "b", 0, "e"]
    sublist = ["c", "d"]

    # test regular replacement
    expected_result = ["a", "b", "c", "d", "e"]
    assert insert_and_remove_placeholder(list_with_placeholder, sublist, 0) == expected_result

    # test boundary cases:
    # replacement at the beginning of the list
    list_with_placeholder = [0, "a", "b", "e"]
    expected_result = ["c", "d", "a", "b", "e"]
    assert insert_and_remove_placeholder(list_with_placeholder, sublist, 0) == expected_result

    # replacement at the end of the list
    list_with_placeholder = ["a", "b", "e", 0]
    expected_result = ["a", "b", "e", "c", "d"]
    assert insert_and_remove_placeholder(list_with_placeholder, sublist, 0) == expected_result

    # multiple occurrences of placeholder; should only be replaced once
    list_with_placeholder = ["a", "b", 0, "e", 0]
    expected_result = ["a", "b", "c", "d", "e", 0]
    assert insert_and_remove_placeholder(list_with_placeholder, sublist, 0) == expected_result

    # faulty input (no list objects or placeholder not in list)
    assert insert_and_remove_placeholder(None, sublist, 0) is None
    assert insert_and_remove_placeholder(list_with_placeholder, None, 0) is None
    assert insert_and_remove_placeholder(list_with_placeholder, sublist, 1) is None

    # replacement with empty sublist
    list_with_placeholder = ["a", "b", "e", 0]
    sublist = []
    expected_result = ["a", "b", "e"]
    assert insert_and_remove_placeholder(list_with_placeholder, sublist, 0) == expected_result


def test_remove_and_insert_placeholder():
    full_list = ["a", "b", "c", "d", "e"]

    # replacement in the middle of the list
    sublist = ["b", "c", "d"]
    expected_result = ["a", 0, "e"]
    assert remove_and_insert_placeholder(full_list, sublist, 0) == expected_result

    # replacement at the beginning of the list
    sublist = ["a", "b", "c"]
    expected_result = [0, "d", "e"]
    assert remove_and_insert_placeholder(full_list, sublist, 0) == expected_result

    # replacement at the end of the list
    sublist = ["c", "d", "e"]
    expected_result = ["a", "b", 0]
    assert remove_and_insert_placeholder(full_list, sublist, 0) == expected_result

    # boundary cases:
    # None inputs
    assert remove_and_insert_placeholder(None, sublist, 0) is None
    assert remove_and_insert_placeholder(full_list, None, 0) == full_list
    assert remove_and_insert_placeholder(full_list, sublist, None) == full_list

    # empty string is a valid placeholder!
    expected_result = ["a", "b", ""]
    assert remove_and_insert_placeholder(full_list, sublist, "") == expected_result

    # full replacement
    sublist = full_list
    assert remove_and_insert_placeholder(full_list, sublist, 0) == [0]

    # sublist is not found in full list
    sublist = ["x", "y", "z"]
    assert remove_and_insert_placeholder(full_list, sublist, 0) == full_list
    sublist = ["a", "b", "z"]
    assert remove_and_insert_placeholder(full_list, sublist, 0) == full_list
