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



