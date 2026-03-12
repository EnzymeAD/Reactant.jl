# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cProfile
import logging
import os
import pstats
import sys
from collections import Counter
from functools import partial
from os import system
from types import SimpleNamespace

# import beanmachine.ppl.compiler.performance_report as pr
from beanmachine.ppl.compiler import performance_report as pr

eprint = partial(print, file=sys.stderr)

from .utils import create_subdir

ppl_profiler = None

DETERMINISTIC_PROFILING = "deterministic"
STATISTICAL_PROFILING = "statistical"
DEFAULT_HOGGERS = 10
LOGGER_NAME = "pplbench"
SIGINT = 2


def start_profiling_BMG(graph):
    # This should never be called before initializing the PPL profiler
    global ppl_profiler
    if ppl_profiler is not None:
        ppl_profiler.start_profiling_BMG(graph)
    return


class PPL_Profiler:
    def __init__(self, output_dir: str, config: SimpleNamespace):
        # list the variables of the object first
        self.LOGGER = logging.getLogger(LOGGER_NAME)  # For output and error messages
        self.profiling = False  # If profiling is enabled or not
        self.output_dir = output_dir  # Output directory for the profiling results
        self.num_profiled = DEFAULT_HOGGERS  # Number of top functions hogging the CPU
        self.strip_profiled_names = False  # Flag to strip the directory names from output, False by default, set to True at YOR
        self.profiling_tools_dir = ""  # Location for the flamegraph PERL script and the smapling profiler py-spy
        self.profiling_type = DETERMINISTIC_PROFILING  # Either determinsitic profiling or statistical, but not both
        self.profiler = None  # To be set later for the Python internal profiler
        self.graph = None  # A pointer to BMG if we are profiling a BMG
        self.bmg_profiling = (
            False  # A flag to indicate the status of the BMG deterministic profiler
        )
        self.statistical_sampler_pid = 0  # The PY-SPY sampler

        global ppl_profiler
        ppl_profiler = self

        self.check_and_set_parameters(output_dir, config)
        if self.profiling:
            create_subdir(self.output_dir, "profile_data")
            self.start_profiling()
        return

    def check_and_set_parameters(self, output_dir: str, config: SimpleNamespace):
        if (not hasattr(config, "profile_run")) or (not config.profile_run):
            self.profiling = False
            return

        # At this point, the user has indicated that this is a profiling run, let us check if the mandatory parameters make sense or not
        if output_dir == "":
            self.LOGGER.error(
                "Illegal output directory in profiler, profiling disabled" + output_dir
            )
            return
        elif not hasattr(config, "profiling_tools_dir"):
            self.LOGGER.error(
                "Profiling tools directory not specified, profiling disabled"
            )
            return

        # All is well, a profiling run with the mandatory parameters available
        self.profiling = True
        self.profiling_tools_dir = config.profiling_tools_dir
        self.output_dir = output_dir

        # At this point, we have a profiling run with a specified output directory and a profiling tools directory. Check the non-mandatory parameters

        self.num_profiled = (
            DEFAULT_HOGGERS
            if not hasattr(config, "num_profiled")
            else config.num_profiled
        )
        self.strip_profiled_names = (
            False
            if not hasattr(config, "strip_profiled_names")
            else config.strip_profiled_names
        )
        self.profiling_type = (
            DETERMINISTIC_PROFILING
            if not hasattr(config, "profiling_type")
            else config.profiling_type
        )
        return

    def start_profiling(self):
        self.LOGGER.info("Starting PPL profiler")
        if self.profiling_type == DETERMINISTIC_PROFILING:
            self.start_deterministic_profiler()
        elif self.profiling_type == STATISTICAL_PROFILING:
            self.start_statistical_profiler()
        else:
            self.LOGGER.error("Unknown profiling method. Exiting")
            exit(1)
        return

    def start_deterministic_profiler(self):
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        return

    def start_statistical_profiler(self):
        my_pid = os.getpid()
        forked_pid = os.fork()
        if forked_pid == 0:
            self.LOGGER.error("Child:: Child sampler has been created")
            executable = self.profiling_tools_dir + "/py-spy"
            command = "record"
            arg0 = "-o"
            output_file = self.output_dir + "/profile_data/flame.svg"
            arg1 = "--native"
            arg2 = "--pid"
            os.execl(
                executable,
                executable,
                command,
                arg0,
                output_file,
                arg1,
                arg2,
                str(my_pid),
            )
            # We should never get to this point
            exit(1)
        else:
            self.statistical_sampler_pid = forked_pid
        return

    def start_profiling_BMG(self, bmgraph):
        if (
            self.bmg_profiling
        ):  # We already started the profiler, no need to redo that again
            pass
        elif self.profiling and self.profiling_type == DETERMINISTIC_PROFILING:
            self.LOGGER.info("Starting BMG profiling")
            self.graph = bmgraph
            self.bmg_profiling = True
            self.graph.collect_performance_data(True)
        return

    def finish_profiling(self):
        if not self.profiling:
            return
        self.LOGGER.info("Stopping PPL profiler")
        if self.profiling_type == DETERMINISTIC_PROFILING:
            self.finish_deterministic_profiling()
        elif self.profiling_type == STATISTICAL_PROFILING:
            self.finish_statistical_profiling()
            exit(1)
        else:
            self.LOGGER.error("Unrecognized profiling type")
            exit(1)
        return

    def finish_statistical_profiling(self):
        os.kill(self.statistical_sampler_pid, SIGINT)
        return

    def print_info(self):  # this function is useful for debugging
        if self.profiling:
            self.LOGGER.info("Printing information about PPL_Profiler")
            self.LOGGER.info("Profiling run: " + str(self.profiling))
            self.LOGGER.info("Output directory is " + self.output_dir)
            self.LOGGER.info("Profiling directory is " + self.profiling_tools_dir)
            self.LOGGER.info("Profiling type is " + self.profiling_type)
            self.LOGGER.info(
                "Number of top functions required " + str(self.num_profiled)
            )
            self.LOGGER.info(
                "Is stripping names required? " + str(self.strip_profiled_names)
            )
        else:
            self.LOGGER.info("Profiler is disabled")
        return

    def finish_deterministic_profiling(self):
        # Stop the profiler
        self.profiler.disable()
        # dump the statistics into the output directoory for general post-processing
        sub_dir = self.output_dir + "/profile_data"
        stat_file = sub_dir + "/profile_stats.pstat"
        self.profiler.dump_stats(stat_file)
        # Print the top functions in CPU usage
        hogger_file = sub_dir + "/top_functions.txt"
        with open(hogger_file, "w") as stream:
            output_stats = pstats.Stats(self.profiler, stream=stream)
            if self.strip_profiled_names:
                output_stats.strip_dirs()
            output_stats.sort_stats("tottime")
            output_stats.print_stats(self.num_profiled)
        # create the file to be used for flame graph presentation
        raw_stat_file_name = sub_dir + "/raw_stat.log"
        self.create_raw_stat_file(output_stats.stats, raw_stat_file_name)
        flame_graph_file_name = sub_dir + "/flame.svg"
        create_flame_graph_command = self.profiling_tools_dir + "/flamegraph.pl"
        create_flame_graph_args = (
            " " + raw_stat_file_name + " > " + flame_graph_file_name
        )
        system(create_flame_graph_command + create_flame_graph_args)

        # Handle the BMG stats if applicable
        if not self.bmg_profiling:
            return
        perf_report = pr.json_to_perf_report(self.graph.performance_report())
        bmg_stat_file_name = sub_dir + "/bmgstat.txt"
        with open(bmg_stat_file_name, "w") as bmg_stat_file:
            bmg_stat_file.write(str(perf_report))

    def create_raw_stat_file(self, stats, raw_stat_file_name):
        log_mult = 1000000
        raw_stat_file = open(raw_stat_file_name, "w")
        functions, calls = self.process_stats(stats)

        blocks = self.prepare(functions, calls)
        for b in blocks:
            trace = []
            for t in b["trace"]:
                trace.append("{}:{}:{}".format(*t))
            print(";".join(trace), round(b["ww"] * log_mult), file=raw_stat_file)
        raw_stat_file.close()

    """ The code below is adapted from the code for FrameProf by Anton Bobrov, copyright (c) 2017 under the MIT license """

    def process_stats(self, stats):
        roots = []
        funcs = {}
        calls = {}
        for func, (cc, nc, tt, ct, clist) in stats.items():
            funcs[func] = {"calls": [], "called": [], "stat": (cc, nc, tt, ct)}
            if not clist:
                roots.append(func)
                calls[("root", func)] = funcs[func]["stat"]

        for func, (_, _, _, _, clist) in stats.items():
            for cfunc, t in clist.items():
                assert (cfunc, func) not in calls
                funcs[cfunc]["calls"].append(func)
                funcs[func]["called"].append(cfunc)
                calls[(cfunc, func)] = t

        total = sum(funcs[r]["stat"][3] for r in roots)
        ttotal = sum(funcs[r]["stat"][2] for r in funcs)

        if not (0.8 < total / ttotal < 1.2):
            eprint(
                "Warning: flameprof can't find proper roots, root cumtime is {} but sum tottime is {}".format(
                    total, ttotal
                )
            )

        # Try to find suitable root
        newroot = max(
            (r for r in funcs if r not in roots), key=lambda r: funcs[r]["stat"][3]
        )
        nstat = funcs[newroot]["stat"]
        ntotal = total + nstat[3]
        if 0.8 < ntotal / ttotal < 1.2:
            roots.append(newroot)
            calls[("root", newroot)] = nstat
            total = ntotal
        else:
            total = ttotal

        funcs["root"] = {"calls": roots, "called": [], "stat": (1, 1, 0, total)}

        return funcs, calls

    def prepare(self, funcs, calls):
        threshold = 0.0001
        blocks = []
        block_counts = Counter()

        def _counts(parent, visited, level=0):
            for child in funcs[parent]["calls"]:
                k = parent, child
                block_counts[k] += 1
                if block_counts[k] < 2:
                    if k not in visited:
                        _counts(child, visited | {k}, level + 1)

        def _calc(
            parent, timings, level, origin, visited, trace=(), pccnt=1, pblock=None
        ):
            children = funcs[parent]["calls"]
            _, _, ptt, ptc = timings
            fchildren = sorted(
                (
                    (
                        f,
                        funcs[f],
                        calls[(parent, f)],
                        max(block_counts[(parent, f)], pccnt),
                    )
                    for f in children
                ),
                key=lambda r: r[0],
            )

            gchildren = [r for r in fchildren if r[3] == 1]

            bchildren = [r for r in fchildren if r[3] > 1]
            if bchildren:
                gctc = sum(r[2][3] for r in gchildren)
                bctc = sum(r[2][3] for r in bchildren)
                rest = ptc - ptt - gctc
                if bctc > 0:
                    factor = rest / bctc
                else:
                    factor = 1
                bchildren = [
                    (
                        f,
                        ff,
                        (
                            round(cc * factor),
                            round(nc * factor),
                            tt * factor,
                            tc * factor,
                        ),
                        ccnt,
                    )
                    for f, ff, (cc, nc, tt, tc), ccnt in bchildren
                ]

            for child, _, (cc, nc, tt, tc), ccnt in gchildren + bchildren:
                if tc / maxw > threshold:
                    ckey = parent, child
                    ctrace = trace + (child,)
                    block = {
                        "trace": ctrace,
                        "color": (pccnt == 1 and ccnt > 1),
                        "level": level,
                        "name": child[2],
                        "hash_name": "{0[0]}:{0[1]}:{0[2]}".format(child),
                        "full_name": "{0[0]}:{0[1]}:{0[2]} {5:.2%} ({1} {2} {3} {4})".format(
                            child, cc, nc, tt, tc, tc / maxw
                        ),
                        "w": tc,
                        "ww": tt,
                        "x": origin,
                    }
                    blocks.append(block)
                    if ckey not in visited:
                        _calc(
                            child,
                            (cc, nc, tt, tc),
                            level + 1,
                            origin,
                            visited | {ckey},
                            ctrace,
                            ccnt,
                            block,
                        )
                elif pblock:
                    pblock["ww"] += tc

                origin += tc

        maxw = funcs["root"]["stat"][3] * 1.0
        _counts("root", set())
        _calc("root", (1, 1, maxw, maxw), 0, 0, set())

        return blocks
