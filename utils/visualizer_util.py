import statistics
import tabulate
import time
from typing import Any, Dict, List

import torch
from torch.fx import Interpreter
from torch.profiler import profile, tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter


class Visualizers:
    def __init__(self, visual_root, savegraphs=True):
        self.writer = SummaryWriter(visual_root)
        self.sub_root = time.strftime('%Y年%m月%d日%H时%M分%S秒', time.localtime())
        self.savegraphs = savegraphs

    def vis_write(self, main_tag, tag_scalar_dict, global_step):
        self.writer.add_scalars(self.sub_root + '_{}'.format(main_tag), tag_scalar_dict, global_step)

    def vis_graph(self, model, input_to_model=None):
        if self.savegraphs:
            with self.writer as w:
                w.add_graph(model, input_to_model)
                self.savegraphs = False

    def vis_image(self, tag, img_tensor, epoch=None, step=None, formats='CHW'):
        if epoch is not None:
            self.writer.add_image(self.sub_root + f'_{tag}_{epoch}', img_tensor, global_step=step, dataformats=formats)
        else:
            self.writer.add_image(self.sub_root + f'_{tag}', img_tensor, global_step=step, dataformats=formats)

    def vis_images(self, tag, img_tensor, epoch=None, step=None, formats='NCHW'):
        if epoch is not None:
            self.writer.add_images(self.sub_root + f'_{tag}_{epoch}', img_tensor, global_step=step, dataformats=formats)
        else:
            self.writer.add_images(self.sub_root + f'_{tag}', img_tensor, global_step=step, dataformats=formats)

    def close_vis(self):
        self.writer.close()


def analysis_profile(model, img_size, save_dir):
    model.eval()
    inputs = torch.randn((1, 3, *img_size))

    with profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=tensorboard_trace_handler(f'{save_dir}/analysis'),
            profile_memory=True,
            record_shapes=True,
            with_stack=True
    ) as profiler:
        start = time.time()
        model(inputs)
        cost = time.time() - start
        print(f"predict_cost = {cost}")
        profiler.step()

    print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print(profiler.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))


# 使用符号跟踪捕获模型，symbolic_trace 有一定编程规范要求
def analysis_graph(model):
    traced_model = torch.fx.symbolic_trace(model)
    print(traced_model.graph)


# 创建分析解释器,调查模型的性能
class ProfilingInterpreter(Interpreter):
    def __init__(self, mod: torch.nn.Module):
        # Rather than have the user symbolically trace their model,
        # we're going to do it in the constructor. As a result, the
        # user can pass in any ``Module`` without having to worry about
        # symbolic tracing APIs
        gm = torch.fx.symbolic_trace(mod)
        super().__init__(gm)

        # We are going to store away two things here:
        #
        # 1. A list of total runtimes for ``mod``. In other words, we are
        #    storing away the time ``mod(...)`` took each time this
        #    interpreter is called.
        self.total_runtime_sec: List[float] = []
        # 2. A map from ``Node`` to a list of times (in seconds) that
        #    node took to run. This can be seen as similar to (1) but
        #    for specific sub-parts of the model.
        self.runtimes_sec: Dict[torch.fx.Node, List[float]] = {}

    ######################################################################
    # Next, let's override our first method: ``run()``. ``Interpreter``'s ``run``
    # method is the top-level entrypoint for execution of the model. We will
    # want to intercept this so that we can record the total runtime of the
    # model.

    def run(self, *args) -> Any:
        # Record the time we started running the model
        t_start = time.time()
        # Run the model by delegating back into Interpreter.run()
        return_val = super().run(*args)
        # Record the time we finished running the model
        t_end = time.time()
        # Store the total elapsed time this model execution took in the
        # ProfilingInterpreter
        self.total_runtime_sec.append(t_end - t_start)
        return return_val

    ######################################################################
    # Now, let's override ``run_node``. ``Interpreter`` calls ``run_node`` each
    # time it executes a single node. We will intercept this so that we
    # can measure and record the time taken for each individual call in
    # the model.

    def run_node(self, n: torch.fx.Node) -> Any:
        # Record the time we started running the op
        t_start = time.time()
        # Run the op by delegating back into Interpreter.run_node()
        return_val = super().run_node(n)
        # Record the time we finished running the op
        t_end = time.time()
        # If we don't have an entry for this node in our runtimes_sec
        # data structure, add one with an empty list value.
        self.runtimes_sec.setdefault(n, [])
        # Record the total elapsed time for this single invocation
        # in the runtimes_sec data structure
        self.runtimes_sec[n].append(t_end - t_start)
        return return_val

    ######################################################################
    # Finally, we are going to define a method (one which doesn't override
    # any ``Interpreter`` method) that provides us a nice, organized view of
    # the data we have collected.

    def summary(self, should_sort: bool = False) -> str:
        # Build up a list of summary information for each node
        node_summaries: List[List[Any]] = []
        # Calculate the mean runtime for the whole network. Because the
        # network may have been called multiple times during profiling,
        # we need to summarize the runtimes. We choose to use the
        # arithmetic mean for this.
        mean_total_runtime = statistics.mean(self.total_runtime_sec)

        # For each node, record summary statistics
        for node, runtimes in self.runtimes_sec.items():
            # Similarly, compute the mean runtime for ``node``
            mean_runtime = statistics.mean(runtimes)
            # For easier understanding, we also compute the percentage
            # time each node took with respect to the whole network.
            pct_total = mean_runtime / mean_total_runtime * 100
            # Record the node's type, name of the node, mean runtime, and
            # percent runtime
            node_summaries.append(
                [node.op, str(node), mean_runtime, pct_total])

        # One of the most important questions to answer when doing performance
        # profiling is "Which op(s) took the longest?". We can make this easy
        # to see by providing sorting functionality in our summary view
        if should_sort:
            node_summaries.sort(key=lambda s: s[2], reverse=True)

        # Use the ``tabulate`` library to create a well-formatted table
        # presenting our summary information
        headers: List[str] = [
            'Op type', 'Op', 'Average runtime (s)', 'Pct total runtime'
        ]
        return tabulate.tabulate(node_summaries, headers=headers)


# 其中使用到了 torch.fx.symbolic_trace()
def analysis_interpreter(model):
    interp = ProfilingInterpreter(model)
    interp.run(input)
    print(interp.summary(True))
