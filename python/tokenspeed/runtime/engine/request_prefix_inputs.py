# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class RequestPrefixInputBuilder:
    """Build bounded request-prefix inputs without owning model state."""

    def __init__(self, lookback: int) -> None:
        self.lookback = max(0, int(lookback))
        self._slot_owners: dict[int, object] = {}

    def gather(self, forward_op, states):
        """Return changed ``(slot, boundary, token_tail)`` inputs."""
        if self.lookback == 0:
            return None

        prefixes = []
        num_extends = forward_op.num_extends()
        for index, (request_id, slot) in enumerate(
            zip(forward_op.request_ids, forward_op.request_pool_indices)
        ):
            slot = int(slot)
            if index >= num_extends and self._slot_owners.get(slot) == request_id:
                continue

            state = states[request_id]
            boundary = (
                int(forward_op.extend_prefix_lens[index])
                if index < num_extends
                else max(0, len(state.prompt_input_ids) + len(state.output_ids) - 1)
            )
            prefixes.append((slot, boundary, self._tail(state, boundary)))
            self._slot_owners[slot] = request_id
        return tuple(prefixes) or None

    def _tail(self, state, boundary: int) -> tuple[int, ...]:
        begin = max(0, boundary - self.lookback)
        prompt = state.prompt_input_ids
        prompt_length = len(prompt)
        if boundary <= prompt_length:
            return tuple(prompt[begin:boundary])
        output_end = boundary - prompt_length
        if begin >= prompt_length:
            return tuple(state.output_ids[begin - prompt_length : output_end])
        return tuple(prompt[begin:] + state.output_ids[:output_end])
