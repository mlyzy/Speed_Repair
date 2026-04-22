from datasets import load_dataset
from datasets import load_from_disk
import pandas as pd
from tqdm import tqdm
from transformers import RobertaTokenizer
import numpy as np
from parser import DFG_java, DFG_csharp, DFG_python
from parser import (
    remove_comments_and_docstrings,
    tree_to_token_index,
    index_to_code_token,
    tree_to_variable_index,
    detokenize_code,
    tree_to_token_nodes,
)
from tree_sitter import Language, Parser
import os
import pickle
import ipdb


def bulid_tree(javacode,tokenize):
    # #ipdb.set_trace()
    code_tokens, ast_leaves, ast_leaf_ranges=add_structure(javacode,tokenize)
    id_tokens = tokenize_codes_texts(javacode)
    # code_ranges = get_code_tokens_ranges(javacode,id_tokens,tokenize)
    # ast_leaf_code_token_idxs=get_leaf_code_token_indices(id_tokens, code_ranges, ast_leaf_ranges)
    sims, lr_paths=get_ast_lr_paths_and_ll_sim(ast_leaves)
    # print(lr_paths)
    # return lr_paths,ast_leaf_code_token_idxs
    return lr_paths


def extract_structure(code, parser):
    # ast
    tree = parser[0].parse(bytes(code, "utf8"))
    root_node = tree.root_node
    ast_token_nodes = tree_to_token_nodes(root_node)  # leaves
    tokens_index = [(node.start_point, node.end_point) for node in ast_token_nodes]
    code = code.split("\n")
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]

    return code_tokens, ast_token_nodes



def add_structure(code, lang="java"):
    LANGUAGE = Language("/mnt/hdd/yzy/yangzhenyu/FastCorrect/FC_utils/parser/my-languages2.so", "c_sharp" if lang == "cs" else lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    dfg_function = {
        "java": DFG_java
    }
    parser = [parser, dfg_function[lang]]

    code_tokens, ast_leaves = extract_structure(
        code, parser
    )
    ast_leaf_ranges=format_node_ranges(code, ast_leaves)
    return code_tokens, ast_leaves, ast_leaf_ranges


def get_code_tokens_ranges(code, id_tokens,tokenize):
    tokenizer = RobertaTokenizer.from_pretrained("/mnt/hdd/yzy/yangzhenyu/pretrain_model/codegraphbert/")
    ranges = []

    code_tokens = [tokenizer.decode(ct) for ct in id_tokens][
        1:-1
    ]  # 1:-1 to remove <s> and </s>
    code2 = "".join(
        code_tokens
    )  # may miss some spaces / special chars that are in row.code_col

    # map each position in code2 to a position in code
    code2_to_code = []
    j = 0
    for i in range(len(code2)):
        while code2[i] != code[j]:
            j += 1
        code2_to_code.append(j)

    # map each code token to a range in code
    code2_idx = 0
    curr_ranges = []
    for ct in code_tokens:
        s, e = code2_idx, code2_idx + len(ct)
        code2_idx = e
        curr_ranges.append((min(code2_to_code[s:e]), 1 + max(code2_to_code[s:e])))
    ranges=[None] + curr_ranges + [None] # first and last for <s> and </s>

    return ranges


def overlap(s1, e1, s2, e2):
    return s1 <= s2 < e1 or s2 <= s1 < e2

def get_leaf_code_token_indices(id_tokens, code_ranges, ast_leaf_ranges):
    ast_leaf_token_idxs = []
    ast_leaf_token_idxs.append([])
    code_tokens_last_idx = len(id_tokens) - 1
    code_tokens_ranges = code_ranges
    for s, e in ast_leaf_ranges:  # s,e为单词的起始位置和终止位置
        if s == e:  # there are leaves with start_point=end_point
            ast_leaf_token_idxs[-1].append([])
            continue
        j = 1
        while not (
            overlap(s, e, code_tokens_ranges[j][0], code_tokens_ranges[j][1])
        ):
            j += 1
            if j == code_tokens_last_idx:  # can't find code tokens for this leaf
                break
        if j == code_tokens_last_idx:  # can't find code tokens for this leaf
            ast_leaf_token_idxs[-1].append([])
            continue
        curr_leaf_token_idxs = []
        while overlap(s, e, code_tokens_ranges[j][0], code_tokens_ranges[j][1]):
            curr_leaf_token_idxs.append(j)
            j += 1
            if j == code_tokens_last_idx:
                break
        ast_leaf_token_idxs[-1].append(curr_leaf_token_idxs)
    ast_leaf_code_token_idxs = ast_leaf_token_idxs
    return ast_leaf_code_token_idxs

def get_ll_sim(p1, p2):
    common = 1
    for i in range(2, min(len(p1), len(p2)) + 1):
        if p1[-i] == p2[-i]:
            common += 1
        else:
            break
    return common * common / (len(p1) * len(p2))

def get_ll_sim(p1, p2):
    common = 1
    for i in range(2, min(len(p1), len(p2)) + 1):
        if p1[-i] == p2[-i]:
            common += 1
        else:
            break
    return common * common / (len(p1) * len(p2))

def get_lr_path(leaf):
    path = [leaf]
    while path[-1].parent is not None:
        path.append(path[-1].parent)
    return path

def get_ast_lr_paths_and_ll_sim(ast_leaves):
    L = min(len(ast_leaves), 512)
    curr_paths = [get_lr_path(leaf) for leaf in ast_leaves]  # 获取此节点的所有父类直到根节点
    curr_sims = np.ones((L, L))
    for i in range(L - 1):
        for j in range(i + 1, L):
            curr_sims[i, j] = curr_sims[j, i] = get_ll_sim(
                curr_paths[i], curr_paths[j]
            )
    sims=curr_sims
    lr_paths=[[node.type for node in path] for path in curr_paths]
    return sims, lr_paths

def get_tokenizer_chars(tokenizer):
    tokenizer_chars = []
    for i in range(tokenizer.vocab_size):
        token = tokenizer.decode(i)
        if len(token) == 1:
            tokenizer_chars.append(token)
    tokenizer_chars = [c for c in tokenizer_chars if c != "�"]
    return tokenizer_chars

def tokenize_codes_texts(texts,tokenize):
    tokenizer = RobertaTokenizer.from_pretrained("/mnt/hdd/yzy/yangzhenyu/pretrain_model/codegraphbert/")
    tokenizer_chars = get_tokenizer_chars(tokenizer)
    texts = "".join(filter(lambda c: c in tokenizer_chars, texts))
    tokenized_texts = tokenizer(texts).input_ids
    return tokenized_texts


def format_node_ranges(code, nodes):
    line_lens = [len(line) + 1 for line in code.split("\n")]
    line_starts = [0] + list(np.cumsum(line_lens))
    return [
        (
            line_starts[node.start_point[0]] + node.start_point[1],
            line_starts[node.end_point[0]] + node.end_point[1],
        )
        for node in nodes
    ]
if __name__ == "__main__":
    ###ipdb.set_trace()
    tree = bulid_tree("public ListSpeechSynthesisTasksResult listSpeechSynthesisTasks(ListSpeechSynthesisTasksRequest request) {request = beforeClientExecution(request);return executeListSpeechSynthesisTasks(request);}")