# Software Becoming Weights

Compiling a job-shop scheduler into transformer weights, so the model executes it token by token — with a trace you can verify.

Arturo Rodrigues
2026
9 min read

## The Problem

There's an increasing gap at the center of applied AI today. On one side: LLMs that explain branch-and-bound in nuanced prose, discuss the theory of job-shop scheduling with impressive fluency, and write working C code for a solver on demand. On the other side: those same models failing to actually solve even a small instance *reliably*. Ask GPT-4 to schedule three jobs on three machines and minimize makespan, and you'll get something that looks right, formats nicely, and is subtly wrong in a way that's *hard to catch*.

The failure mode is structural, not a tuning problem. Hao et al. showed at ICLR 2026 that simply renaming variables to α and β was enough to degrade model performance on reasoning tasks: the model is keying off surface-level patterns, not executing the algorithm. For problems that are **NP-hard**, LLMs are effectively guessing — there's no execution trace, no backtracking, no way to verify the answer is correct. For a research demo, a wrong answer is embarrassing. In a factory, it means misallocated machines, missed deadlines, and lost margin.

> **The question**
>
> What if, instead of asking an LLM to *describe* how to run a scheduler, we could compile the scheduler directly into the transformer weights and let the model *execute* it step by step — producing a verifiable, token-by-token trace?

I came across Percepta's [transformer-vm](https://github.com/Percepta-Core/transformer-vm), which compiles a WebAssembly VM analytically into standard transformer weights. As a result, any C program you can compile to WASM will execute inside the model's forward pass, token by token, with byte-identical output to the native binary.

I forked Percepta's library and applied it to one of the most famous NP-hard problems in manufacturing: the Job-Shop Scheduling Problem (JSSP). But first, the key theoretical points.

## 01 — Foundations: Theoretical Background

The key architectural insight is that a transformer has exactly what you need to run a computer:

* **Attention heads implement keyed memory lookup.** Percepta encodes each integer key *k* as the 2-D point (2k, −k²) and queries with direction (q, 1). The attention score becomes 2qk − k² = −(k − q)² + q², which is uniquely maximized at exactly k = q. One 2-D attention head, one exact lookup.
* **Feed-forward layers implement exact products and conditionals.** The identity ReLU(z + 1) − ReLU(z) is an exact step function for integer inputs — which means you can build equality tests, conditionals, and arbitrary integer arithmetic from neurons alone.
* **The residual stream acts as addressable RAM** — a running vector that each layer reads from and writes to.

The five primitives this gives you — Read/Write, Cumulative Sum, Product, Conditional, and Linear Combination — are sufficient for Turing-complete computation. Percepta's compiler takes a WASM program, expresses it as a gate graph of these primitives, and uses a Mixed-Integer Linear Program to schedule those gates into transformer layers and pack values into residual-stream slots. The result is a set of weight matrices that, run autoregressively, execute the program exactly.

The engineering payoff is an **exponentially faster attention mechanism**. Standard softmax attention costs O(t) per step (scoring the new query against all cached keys). Because the 2-D keys lie on a downward parabola, the highest-scoring key is always a vertex of the convex hull of all key points. Percepta's `HullKVCache` maintains that hull dynamically, answering each lookup in O(log t) instead — turning total generation cost from O(T²) to O(T log T). This is what makes running millions of execution steps tractable.

Their examples run the world's hardest Sudoku, min-cost matching via the Hungarian algorithm, and basic arithmetic. I wanted to add something closer to where I think this technology matters: **production scheduling**.

## 02 — The problem: The Job-Shop Scheduling Problem

The [Job-Shop Scheduling Problem](https://en.wikipedia.org/wiki/Job-shop_scheduling) (JSSP) is the canonical NP-hard problem in advanced manufacturing. Every automotive line, semiconductor fab, chemical batch process, and logistics hub solves a variant of it every single day.

The setup: you have n jobs and m machines. Each job is a fixed sequence of operations — a "routing" — where each operation requires a specific machine for a given duration. The constraints are strict: a machine processes one operation at a time, a job's operations must run in order, and there's no preemption (you can't interrupt an operation midway). The goal is to minimize the **makespan**: the time from when the first operation starts to when the last one finishes.

Even a 3 × 3 instance (3 jobs, 3 machines, 9 operations) has a branching factor that produces a search tree large enough to require pruning. A 10 × 10 instance — the standard hard benchmark — has been studied for decades. The JSSP resists greedy solutions; local optima are dense and misleading. Getting the optimal schedule requires systematic search. This is precisely the kind of exact, backtracking computation that LLMs cannot perform reliably and that a compiled transformer can.

## 03 — The instance: Three Jobs, Three Machines

Our 3 × 3 problem:

- **Job 0**: M0 (3) → M1 (2) → M2 (2), work = 7
- **Job 1**: M0 (2) → M2 (1) → M1 (4), work = 7
- **Job 2**: M1 (4) → M2 (3) → M0 (1), work = 8

The lower bound on makespan comes from two observations:

1. **Max job work.** Job 2 has 8 units of total processing. No schedule can complete it faster.
2. **Max machine load.** M1 must process Job 0's op1 (2), Job 1's op2 (4), and Job 2's op0 (4) — 10 units total. A machine can't work in parallel with itself, so makespan ≥ 10.

The lower bound is therefore **10**. Whether a feasible schedule exists at that bound, or whether precedence constraints push it higher, is exactly what the solver determines. You cannot answer this by staring at the numbers — you have to run the search.

The solver uses depth-first branch-and-bound. At each node it maintains:

* `op_done[j]` — how many of job j's operations are scheduled so far;
* `job_ready[j]` — when job j can next start work (the end time of its last scheduled op);
* `mach_ready[m]` — when machine m is next available.

**Branch.** At each depth, try scheduling the next eligible operation from every job that still has work. For each candidate, compute the earliest feasible start (the max of the job's readiness and the machine's availability), place the operation, recurse, then undo and try the next.

**Prune.** Before recursing, check whether the current maximum `mach_ready` across all machines already meets or exceeds the best makespan found so far. If so, no completion of this branch can improve things. Cut.

The recursion reaches depth 9 (one level per operation) with branching factor at most 3 (one per job). Pruning collapses the effective tree dramatically.

## 04 — The code: Static Arrays, No Heap

The runtime imposes three constraints that shape everything: no heap allocation, integers only, and no runtime multiplications in hot paths. These aren't limitations — for a fixed-size scheduling problem they're the right defaults. All state lives in static arrays the WASM lowering pass can see at compile time. The instance is two tables: where each operation runs, and how long it takes.

```c
static const unsigned char route_m[NJ][NM] = {
    {0, 1, 2},   /* Job 0:  M0 → M1 → M2 */
    {0, 2, 1},   /* Job 1:  M0 → M2 → M1 */
    {1, 2, 0},   /* Job 2:  M1 → M2 → M0 */
};
static const unsigned char dur[NJ][NM] = {
    {3, 2, 2},   /* Job 0 durations */
    {2, 1, 4},   /* Job 1 durations */
    {4, 3, 1},   /* Job 2 durations */
};

/* Search state — three static arrays, no malloc */
static unsigned char op_done[NJ];     /* ops scheduled per job     */
static unsigned char job_ready[NJ];   /* when each job is free     */
static unsigned char mach_ready[NM];  /* when each machine is free */
```

The core of the search is `recurse(depth)`. Two things happen at each node: a pruning check that cuts dead branches early, and a branch loop that tries placing the next op from each eligible job, recurses, then undoes the placement.

```c
/* Prune: max(mach_ready) is a valid lower bound — */
/* any future op can only push that value upward.   */
int lb = mach_ready[0];
if (mach_ready[1] > lb) lb = mach_ready[1];
if (mach_ready[2] > lb) lb = mach_ready[2];
if (lb >= best_makespan) { prunes = prunes + 1; return; }

/* Branch: try the next op from each job with work left */
for (j = 0; j < NJ; j = j + 1) {
    int k     = op_done[j];          /* index of j's next op */
    if (k >= NM) continue;
    int m     = route_m[j][k];       /* required machine     */
    int start = job_ready[j];
    if (mach_ready[m] > start) start = mach_ready[m];
    int end   = start + dur[j][k];

    /* apply → recurse → undo */
    int saved_jr = job_ready[j], saved_mr = mach_ready[m];
    op_done[j] = k + 1;  job_ready[j] = end;  mach_ready[m] = end;
    recurse(depth + 1);
    op_done[j] = k;  job_ready[j] = saved_jr;  mach_ready[m] = saved_mr;
}
```

Two details worth noting. First, the increment style: `i = i + 1`, not `i++`. The WASM lowering pass handles constant additions cleanly; post-increment generates different bytecode the runtime handles less efficiently. Second, the `printf()` calls throughout the file aren't just logging — each one emits output tokens. **That stream is the execution trace.**

## 05 — Running it: Compile, Schedule, Execute

```bash
# Add job_shop.c to examples/ and register it in examples/manifest.yaml
uv run wasm-run

# …or bake the program into the weights (the first Futamura projection)
uv run wasm-specialize examples/job_shop.c
```

`wasm-run` compiles the C to WASM, solves the MILP that schedules gate operations into transformer layers, constructs the weight matrices, and runs the model autoregressively. The specialized variant goes one step further: the program itself disappears from the input and lives in the FFN weights instead. The scheduler's instruction table — routing, durations, branch logic — becomes model parameters. **Software becoming weights.**

## 06 — The trace: The Execution Trace

```
3x3 job-shop scheduling (branch-and-bound, minimize makespan)
instance:
  job 0: (M0,3) -> (M1,2) -> (M2,2)
  job 1: (M0,2) -> (M2,1) -> (M1,4)
  job 2: (M1,4) -> (M2,3) -> (M0,1)
lower bound: 10 (max machine total work)
searching...
new best makespan: 16
new best makespan: 14
new best makespan: 13
new best makespan: 11
nodes explored: 118
prunes: 109
optimal makespan: 11
schedule (in scheduler placement order):
J2 op0 on M1: [0, 4]     J0 op0 on M0: [0, 3]
J1 op0 on M0: [3, 5]     J0 op1 on M1: [4, 6]
J1 op1 on M2: [5, 6]     J1 op2 on M1: [6, 10]
J2 op1 on M2: [6, 9]     J2 op2 on M0: [9, 10]
J0 op2 on M2: [9, 11]
```

That trace is the proof. The schedule found places each operation in order with the following execution details:

Three things are worth reading in that output:

* The lower bound of 10 isn't achievable. Job 1's last op (4 units on M1) can't start until its M2 operation finishes, and M2 is blocked by Job 2 at exactly the moment it's needed. Precedence forces the true optimum to **11**. Finding that gap requires genuine search, not estimation.
* Pruning fires on **92%** of nodes visited (109 of 118) — the bound is tight and the tree collapses fast.
* The schedule is verifiable by inspection. M1 runs back-to-back from 0 to 10 with no idle time; every job's routing is respected. This isn't a high-probability guess — it's a certificate.

## 07 — Why it matters: The Line Starts to Dissolve

The standard options for production scheduling are a commercial OR solver (fast, black-box, expensive, hard to customize) or a heuristic (fast, approximate). What's been missing is exact, auditable scheduling computation living inside the same stack as the language model.

That's the opening this creates. When the model needs to reason about a schedule — *"why was Job 1 delayed?"* — the evidence is in the same token stream as the computation. When a constraint changes — *"M1 is down until Tuesday"* — you update the C program, recompile to WASM, get new weights. The line between the model that *explains* the schedule and the model that *runs* the scheduler starts to dissolve.

The Percepta team is explicit that this is a proof-of-concept — slower than a conventional solver, with partial WASM support. But the demonstration is real: a transformer with analytically-computed weights executing branch-and-bound for job-shop scheduling, with verifiable output. The question isn't whether this works. The trace above is the proof. The question is how far it scales.

## 08 — Further reading: References

* Percepta — [Can LLMs Be Computers?](https://www.percepta.ai/blog/can-llms-be-computers)
* Percepta — [Constructing an LLM-Computer](https://www.percepta.ai/blog/constructing-llm-computer)
* Source — [Percepta-Core/transformer-vm](https://github.com/Percepta-Core/transformer-vm) on GitHub

Yuren Hao, Xiang Wan & ChengXiang Zhai. "An Investigation of Robustness of LLMs in Mathematical Reasoning: Benchmarking with Mathematically-Equivalent Transformation of Advanced Mathematical Problems." [ICLR 2026 Workshop on Logical Reasoning of LLMs.](https://openreview.net/forum?id=b8GuXlLc1y)

∎

---

*The code for this essay lives in [`transformer_vm/examples/job_shop.c`](transformer_vm/examples/job_shop.c). The upstream Percepta-Core project documentation has been preserved in [README-upstream.md](README-upstream.md).*
