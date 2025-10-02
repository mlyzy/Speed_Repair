# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import Counter
import pandas as pd
import numpy as np
import torch
from fairseq.file_io import PathManager
from fairseq.tokenizer import tokenize_line
from transformers import RobertaTokenizer, AutoTokenizer, BertTokenizer
import ipdb
from bulidtree import bulid_tree
import pickle
import math


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


class Binarizer:
    @staticmethod
    def binarize(
        filename,
        dictfair,
        consumer,
        tokenize=tokenize_line,
        append_eos=True,
        reverse_order=False,
        offset=0,
        end=-1,
        already_numberized=False,
        src_with_werdur=False,
    ):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dictfair.unk_index and word != dictfair.unk_word:
                replaced.update([word])

        # ipdb.set_trace()
        # num=0
        with open(PathManager.get_local_path(filename), "r", encoding="utf-8") as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            # print(len(line))
            node_list = []
            while line:
                # ipdb.set_trace()
                line = line.strip()
                # num=num+1
                if end > 0 and f.tell() > end:
                    break
                if already_numberized:
                    assert (
                        " |||| " not in line
                    ), "This constraint is add when doing asr correction exp"
                    id_strings = line.split()
                    id_list = [int(id_string) for id_string in id_strings]
                    if reverse_order:
                        id_list.reverse()
                    if append_eos:
                        id_list.append(dictfair.eos())
                    ids = torch.IntTensor(id_list)
                else:
                    if " |||| " in line:
                        assert src_with_werdur
                        line, werdur_info = line.split(" |||| ")
                        # file2.write(str(line)+"\n")
                        werdur_list = []
                        for i in werdur_info.split():  # strip()是用于消除字符串前后的给定字符，默认为换行和空格
                            assert abs(int(i)) < 30000
                            werdur_list.append(int(i) + 32768)
                        if append_eos:
                            werdur_list.append(1 + 32768)
                        werdur_list_length = len(werdur_list)
                    else:
                        # ipdb.set_trace()
                        # astpath, code_token_idxs= bulid_tree(line)

                        astpath = bulid_tree(line)
                        # idxs_lenth = []
                        # for i in range(len(code_token_idxs[0])):
                        #     idxs_lenth.append(len(code_token_idxs[0][i]))
                        node_types = sum(astpath, [])
                        node_list = node_list + node_types
                        # ipdb.set_trace()
                        node_list = list(dict.fromkeys(node_list))
                        node_type_to_ind = {t: i for i, t in enumerate(node_list)}
                        astpath_id = []
                        for i in range(len(astpath)):
                            onepath = []
                            for j in range(len(astpath[i])):
                                onepath.append(node_type_to_ind[astpath[i][j]])
                            astpath_id.append(onepath)
                        # ipdb.set_trace()
                        # for i in range(len(astpath_id)):
                        #     for j in range(len(astpath_id[i])):
                        #         if astpath_id[i][j]>88:
                        #             print(astpath_id[i][j])
                        werdur_list = None
                    # ipdb.set_trace()
                    # Tokenizer = RobertaTokenizer.from_pretrained(
                    #     "/mnt/hdd/yzy/yangzhenyu/pretrain_model/codegraphbert/"
                    # )

                    # tokens = Tokenizer.tokenize(line)
                    # tokens = tokens + [Tokenizer.sep_token]
                    # ids = Tokenizer.convert_tokens_to_ids(tokens)
                    # ids = torch.tensor(ids, dtype=torch.int32)
                    # ipdb.set_trace()
                    ids = dictfair.encode_line(
                        line=line,
                        line_tokenizer=tokenize,  # 空格分词器
                        add_if_not_exist=False,
                        consumer=replaced_consumer,
                        append_eos=append_eos,
                        reverse_order=reverse_order,
                    )
                    # if werdur_list_length != len(ids):
                    #     print("------------------------------------")
                    #     file2.write("yangzhenyu"+"\n")
                    if werdur_list is not None:
                        # if werdur_list_length != len(ids):
                        #     print("------------------------------------")
                        #     print("werdur",werdur_info)
                        #     print("ids",ids)
                        #     print(line)
                        #     print(num)
                        #     print("yangzhenyutest"+"\n")
                        assert werdur_list_length == len(ids)
                        ids = torch.cat([ids, torch.IntTensor(werdur_list)], dim=-1)
                    else:
                        # ipdb.set_trace()
                        # new_lst = [element for sublist in code_token_idxs[0] for element in sublist]
                        # assert len(ids) == (new_lst[-1]+1)
                        # curr_paths = np.array([[-1] * (len(ids) - len(path)) + path for path in astpath_id])
                        curr_paths = [num for row in astpath_id for num in row]
                        path_lenth = [len(path) for path in astpath_id]
                        num = math.ceil(len(curr_paths) / len(ids))
                        try:
                            curr_paths = np.array(
                                [[-1] * (num * len(ids) - len(curr_paths)) + curr_paths]
                            ).reshape(num, -1)
                        except:
                            print(curr_paths)
                            print(line)
                            print(astpath)
                        path_lenth = np.array(
                            [-1] * (len(ids) - len(path_lenth)) + path_lenth
                        )
                        # new_lst = np.array([-1] * (len(ids) - len(new_lst)) + new_lst)
                        # idxs_lenth = np.array([-1] * (len(ids) - len(idxs_lenth)) + idxs_lenth)
                        nodelenth = np.array([-1] * (len(ids) - 1) + [len(ids)])
                        ids = torch.cat(
                            [ids.unsqueeze(0), torch.Tensor(curr_paths)], dim=0
                        )
                        ids = torch.cat(
                            [ids, torch.Tensor(path_lenth).unsqueeze(0)], dim=0
                        )
                        # ids = torch.cat([ids, torch.Tensor(new_lst).unsqueeze(0)], dim=0)
                        # ids = torch.cat([ids, torch.Tensor(idxs_lenth).unsqueeze(0)], dim=0)
                        ids = torch.cat(
                            [ids, torch.Tensor(nodelenth).unsqueeze(0)], dim=0
                        )
                nseq += 1
                ntok += len(ids)
                consumer(ids)
                line = f.readline()
        # ipdb.set_trace()
        file_path = (
            "/mnt/hdd/yzy/yangzhenyu/FastCorrect/data/defect4jnb/all_node_types.pkl"
        )
        if os.path.exists(file_path):
            ft_node_types = pickle.load(open(file_path, "rb"))
            node_list = ft_node_types + node_list
            node_list = list(dict.fromkeys(node_list))

        if len(node_list) != 0:
            pickle.dump(node_list, open(file_path, "wb"))
            # print(node_list)
        # ipdb.set_trace()
        return {
            "nseq": nseq,
            "nunk": sum(replaced.values()),
            "ntok": ntok,
            "replaced": replaced,
        }

    @staticmethod
    def binarize_alignments(filename, alignment_parser, consumer, offset=0, end=-1):
        nseq = 0

        with open(PathManager.get_local_path(filename), "r") as f:
            f.seek(offset)
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                ids = alignment_parser(line)
                nseq += 1
                consumer(ids)
                line = f.readline()
        return {"nseq": nseq}

    @staticmethod
    def find_offsets(filename, num_chunks):
        with open(PathManager.get_local_path(filename), "r", encoding="utf-8") as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_chunks
            offsets = [0 for _ in range(num_chunks + 1)]
            for i in range(1, num_chunks):
                f.seek(chunk_size * i)
                safe_readline(f)
                offsets[i] = f.tell()
            return offsets
