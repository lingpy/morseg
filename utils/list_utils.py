def list_contains_sublist(full_list: list, partial_list: list):
    len_full_list = len(full_list)
    len_partial_list = len(partial_list)

    if len_full_list < len_partial_list:
        return False

    for i in range(len_full_list - len_partial_list + 1):
        if full_list[i:i+len_partial_list] == partial_list:
            return True

    return False


def get_start_idx_for_sublist(full_list: list, partial_list: list):
    len_full_list = len(full_list)
    len_partial_list = len(partial_list)

    if len_full_list < len_partial_list:
        return -1

    for i in range(len_full_list - len_partial_list + 1):
        if full_list[i:i + len_partial_list] == partial_list:
            return i

    return -1


def remove_and_insert_placeholder(full_list: list, partial_list: list, placeholder):
    start_index = get_start_idx_for_sublist(full_list, partial_list)
    end_index = start_index + len(partial_list)

    if start_index != -1 and end_index <= len(full_list):
        modified_list = full_list[:start_index] + [placeholder] + full_list[end_index:]
        return modified_list

    return None


def insert_and_remove_placeholder(full_list: list, partial_list: list, placeholder):
    index = full_list.index(placeholder)
    full_list.remove(placeholder)
    full_list[index:index] = partial_list

    return full_list


def remove_repeating_symbols(l, symbol):
    """
    a method to remove subsequent repetitions of a given symbol from a list
    :param l: the list to be modified
    :param symbol: the symbol to be stripped
    :return: the sanitized list
    """
    modified_list = []

    for i, x in enumerate(l):
        if i == len(l)-1 or x != symbol or x != l[i+1]:
            modified_list.append(x)

    return modified_list


if __name__ == "__main__":
    l = ["a", "b", "+", "c", "+", "+", "+", "+"]
    clean_list = remove_repeating_symbols(l, "+")

    print(clean_list)
