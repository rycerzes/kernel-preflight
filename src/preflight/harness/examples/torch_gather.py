"""Row gather in eager torch: table[idx].

The reference point, and for once close to optimal -- torch's index_select is a real
kernel. The number it reaches is well below peak bandwidth, and that is the operation
rather than the implementation: a random row gather cannot stream.
"""


def launch_candidate(inputs, out, meta):
    out.copy_(inputs["table"][inputs["idx"]])
