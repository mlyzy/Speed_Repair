from fairseq.tokenizer import tokenize_line
from bulidtree import bulid_tree


class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m = len(word1)
        n = len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            dp[i][0] = i

        for j in range(1, n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1

        return dp

    def backpath(self, word1, word2, dp):
        i = len(dp) - 1
        j = len(dp[0]) - 1
        res = []
        while i > 0 or j > 0:
            a = dp[i - 1][j - 1] if i > 0 and j > 0 else float("inf")
            b = dp[i - 1][j] if i > 0 else float("inf")
            c = dp[i][j - 1] if j > 0 else float("inf")
            min_val = min([a, b, c])

            if dp[i][j] == a and a == min_val:
                i -= 1
                j -= 1
                # 没有操作
                res.append((i, word1[i], word2[j], "no change"))

            elif b == min([a, b, c]):
                i = i - 1
                res.append((i, word1[i], "", "del"))
            elif a == min([a, b, c]):
                #  通过替换来的
                i -= 1
                j -= 1
                res.append((i, word1[i], word2[j], "sub"))
            else:
                j = j - 1
                res.append((i, "", word2[j], "ins"))
        # print(res)
        return res


filefixed = open(
    "./dataset/defect4j/buggy.txt",
    "r",
)
filebug = open(
    "./data/defect4j/fixed.txt",
    "r",
)
bug = filebug.readlines()
fixed = filefixed.readlines()
sentnum = len(bug)
f = open(
    r"./data/defect4j/newtrainbuggy.txt",
    "a",
    encoding="UTF-8",
)
f2 = open(
    r"./data/defect4j/newtrainfixed.txt",
    "a",
    encoding="UTF-8",
)

for ii in range(sentnum):
    obj = Solution()
    bugsent = bug[ii].strip()
    fixedsent = fixed[ii].strip()
    # print(len(bugsent))
    if bugsent[-1] == " ":
        bugsent = bugsent[:-1]
    if fixedsent[-1] == " ":
        fixedsent = fixedsent[:-1]
    fixedsent = fixedsent.replace(" 	 ", "")
    bugword = tokenize_line(bugsent)
    bugwlen = len(bugword)
    tree = bulid_tree(fixedsent)
    fixedword = tokenize_line(fixedsent)
    if len(tree) != len(fixedword):
        # print(ii)
        continue
    for w in range(len(fixedword)):
        fixedword[w] = fixedword[w].replace("Ġ", "")
    dp = obj.minDistance(bugword, fixedword)
    res = obj.backpath(bugword, fixedword, dp)
    l = [0 for i in range(len(bugword) + 1)]
    for w in range(len(res) - 1, -1, -1):
        l[res[w][0]] = l[res[w][0]] + 1
    # ipdb.set_trace()
    if l[-1] != 0:
        l[-2] = l[-2] + l[-1]
        l = l[:-1]
    bugstr = bugsent + " ||||"
    idetlen = 0
    pad = 0
    for j in range(len(l)):
        if l[j] > 1:
            bugstr = bugstr + " " + str(-1 * l[j])
            pad = pad + l[j] - 1
            idetlen = idetlen + 1
        if l[j] == 1:
            if res[len(res) - j - pad - 1][3] == "no change":
                bugstr = bugstr + " 1"
                idetlen = idetlen + 1
            if res[len(res) - j - pad - 1][3] == "del":
                bugstr = bugstr + " 0"
                idetlen = idetlen + 1
            if res[len(res) - j - pad - 1][3] == "sub":
                bugstr = bugstr + " -1"
                idetlen = idetlen + 1
    # ipdb.set_trace()
    # print(bugstr)
    if idetlen != bugwlen:
        print(idetlen, bugwlen)
        print("error")
    f.write(bugstr + "\n")
    f2.write(fixedsent + "\n")
