import datargs
from dataclasses import dataclass
import json
from typing import Sequence, Callable
from collections import defaultdict
import io
import gc


class TrieNode(defaultdict):
    def __init__(self):
        super().__init__(TrieNode)
        self.under = 0

    def first_kv(self):
        k = next(iter(self))
        return k, self[k]

    def is_single_child(self):
        return len(self) == 1

    def compress(self, root=False):
        if root or not self.is_single_child():
            for ch, child in list(self.items()):
                match child.compress():
                    case (key, child):
                        del self[ch]
                        self[ch + key] = child
                    case None:
                        pass
                    case _:
                        raise ValueError("Expected compressed child")
            return None

        sb = io.StringIO()
        parent = self
        while parent.is_single_child():
            original_under = parent.under
            ch, descendant = parent.first_kv()
            assert descendant.under == original_under
            # del parent[ch]
            sb.write(ch)
            parent = descendant

        assert parent != self
        assert parent.compress() is None

        gc.collect()
        ret = sb.getvalue()
        return ret, parent

    def dump_compressed(self, prefix: str = "", same_line: bool = False):
        for ch, child in self.items():
            if len(self) == 1:
                p = prefix + f"[{child.under}] "
                word = ch
                if same_line:
                    word = word.replace("\n", "\\n")
                else:
                    word = json.dumps(ch)
                yield (p if not same_line else "") + word + " " + next(child.dump(prefix, same_line=True), "")
            else:
                yield prefix + f"[{child.under}] " + json.dumps(ch)
                yield from child.dump(prefix + "  ")

    def dump(
        self, prefix: str = "", node_len: int | None = None, key_func: Callable | None = None
    ):
        for ch, child in self.items():
            out = None
            if key_func is not None:
                out = key_func(ch)
            elif len(ch) > node_len:
                node_len -= 10
                out = json.dumps(ch[: node_len // 2]) + f"...{len(ch):5d}..." + json.dumps(ch[-node_len // 2 :])
            else:
                out = json.dumps(ch)
            yield prefix + f"[{child.under}] " + out
            yield from child.dump(prefix + "  ", node_len=node_len, key_func=key_func)


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def add(self, s: str):
        current = self.root
        for ch in s:
            current.under += 1
            current = current[ch]

        current.under += 1
        current = current[""]
        current.under += 1

    def compress(self):
        return self.root.compress(root=True)

    def dump(self, node_len: int | None = None, key_func: Callable | None = None):
        return "\n".join(self.root.dump(node_len=node_len, key_func=key_func))


@dataclass
class Config:
    dataset_file: str = datargs.arg(
        help="Path to jsonl dataset file to analyze",
    )
    row_count: int = datargs.arg(
        help="Number of rows to take from the dataset",
        default=100000,
    )
    roles: Sequence[str] = datargs.arg(
        help="[repeated] Only include messages from these roles that ocurr as a prefix of the conversation, "
        "and exclude the rest, after the first non-matching message.",
        default=("system", "user"),
    )


def _get_rows(dataset_file: str, row_count: int | None):
    with open(dataset_file, "r") as f:
        for _, line in zip(range(row_count), f):
            yield json.loads(line)
            del line


def _concatenate_messages(row: dict, roles: Sequence[str]) -> str:
    return "".join(message["content"] for message in row["messages"] if message["role"] in roles)


def main():
    config = datargs.parse(Config)

    trie = Trie()

    for row in _get_rows(config.dataset_file, config.row_count):
        row = _concatenate_messages(row, config.roles)
        trie.add(row)
    trie.compress()

    print(trie.dump(key_func=lambda x: f"len={len(x)}"))
    print()


if __name__ == "__main__":
    main()
