/*
  =============================================================================
  job_shop.c -- Job-shop scheduling, 3x3 instance, branch-and-bound.
  =============================================================================

  WHAT THIS IS
    Solves a tiny job-shop scheduling problem (3 jobs, 3 machines) by
    depth-first branch-and-bound, minimizing makespan (the time when the
    last job finishes). The transformer-VM runtime executes this program
    step-by-step inside transformer weights; the chain-of-thought printf()
    output IS the visible execution trace.

  WHY JOB-SHOP
    Job-shop scheduling is the canonical NP-hard problem in advanced
    manufacturing -- assigning operations to machines under fixed routings,
    minimizing total time. Every shop floor, every chemical batch line,
    every fab solves a variant of this every day. It's a natural test of
    whether a deterministic in-model executor can serve as the substrate
    for a deployable industrial decision system.

  STYLE NOTE FOR THE READER
    Per the author's normal convention, comments explain WHY not WHAT. For
    THIS file (a teaching artifact) we deviate: we also annotate WHAT each
    non-obvious C idiom is doing, so the file is walkable line-by-line.

  CONSTRAINTS THE TRANSFORMER-VM IMPOSES
    - Heap-free: all state in static arrays (no malloc -- the WASM lowering
      pass and runtime have no allocator).
    - Integer-only: no floats anywhere.
    - Avoid native mul/div in hot loops (they are lowered to long add/sub
      sequences). Constant multiplications like `j * 3` are fine -- the
      lowering pass expands them once at compile time.
    - Entry point is compute(const char *input), per the repo convention.
    - Output goes to printf(), which the runtime prints token-by-token.

  INSTANCE
    Three jobs, three machines. Each job has a fixed routing -- a sequence
    of (machine, processing_time) operations that must run in order.

      Job 0:  (M0, 3)  ->  (M1, 2)  ->  (M2, 2)        total work = 7
      Job 1:  (M0, 2)  ->  (M2, 1)  ->  (M1, 4)        total work = 7
      Job 2:  (M1, 4)  ->  (M2, 3)  ->  (M0, 1)        total work = 8

    Lower bound on makespan:
      - Max job total-work = 8 (Job 2)
      - Max machine total-work: M0=6, M1=10, M2=6  -> 10 (Machine 1)
      - So makespan >= 10. The solver finds the true optimum.

  ALGORITHM
    State at each node in the search tree:
      op_done[j]     = how many ops of job j have already been scheduled
      job_ready[j]   = the time job j's last scheduled op finishes
                       (= 0 if no ops scheduled yet)
      mach_ready[m]  = the time machine m becomes free
                       (= 0 if idle so far)

    At each node we BRANCH on which eligible op to schedule next:
      For each job j that still has ops remaining (op_done[j] < NM):
        - k = op_done[j]                  // index of j's next op
        - m = route_m[j][k]               // which machine it needs
        - d = dur[j][k]                   // how long it takes
        - start = max(job_ready[j], mach_ready[m])    // earliest legal start
        - end   = start + d
        - Record the placement, recurse, then undo.

    PRUNE: if the maximum machine_ready already meets or exceeds the best
    makespan found so far, this branch cannot improve -- skip it.

    The recursion is DFS to depth NJ * NM = 9, with branching factor <= NJ.
    Worst-case exploration is bounded; for a 3x3 instance the tree is tiny.

  EXPECTED OUTPUT
    The program prints:
      1. The instance, formatted human-readably.
      2. Each new best makespan as the search discovers it.
      3. Final stats: nodes explored, prunes performed, optimal makespan.
      4. The optimal schedule in placement order: which op runs on which
         machine, with [start, end] times.

    A REFERENCE block at the bottom of this file gives the expected
    optimal makespan and schedule for hand-verification once you've run it.
  =============================================================================
*/


/* ---------- Problem-size constants. ------------------------------------- */
/* NJ = number of jobs. NM = number of machines (equal because this is a
   square instance; in a general M x N shop they would differ).            */
#define NJ 3
#define NM 3

/* TOTAL_OPS is just NJ * NM but written explicitly so we don't have to
   trust the lowering pass to fold the constant in every reference.        */
#define TOTAL_OPS 9

/* INF_MAKESPAN is a sentinel "no schedule found yet" value. Any real
   makespan on a 3x3 instance fits comfortably under 255 (total work is
   below 25), so 255 is a safe upper bound that still fits in unsigned
   char arithmetic everywhere it's compared.                                */
#define INF_MAKESPAN 255


/* ---------- Instance data (hard-coded, hand-verifiable). ---------------- */

/* route_m[j][k] = which machine the k-th operation of job j runs on.
   Two-dimensional access (route_m[j][k]) is fine: the compiler resolves
   the row offset as j*NM, and NM is a small constant, so the WASM
   lowering pass expands it to a short add-chain at compile time.          */
static const unsigned char route_m[NJ][NM] = {
    {0, 1, 2},   /* Job 0:  M0 then M1 then M2                              */
    {0, 2, 1},   /* Job 1:  M0 then M2 then M1                              */
    {1, 2, 0},   /* Job 2:  M1 then M2 then M0                              */
};

/* dur[j][k] = processing time of the k-th operation of job j on its
   designated machine. Same indexing convention as route_m.                 */
static const unsigned char dur[NJ][NM] = {
    {3, 2, 2},   /* Job 0 durations                                         */
    {2, 1, 4},   /* Job 1 durations                                         */
    {4, 3, 1},   /* Job 2 durations                                         */
};


/* ---------- Mutable search state (static = no heap, zero-init). --------- */

/* op_done[j] is how many of job j's operations have been scheduled so far
   in the CURRENT partial schedule. Range 0..NM. When op_done[j] == NM
   the job is fully scheduled.                                              */
static unsigned char op_done[NJ];

/* job_ready[j] is the earliest time job j can START its next operation
   (= the end time of its most recently scheduled op, or 0 if none yet).
   Each job is a linear chain: ops must run in routing order, so an op
   cannot start before its predecessor finishes.                            */
static unsigned char job_ready[NJ];

/* mach_ready[m] is the earliest time machine m is FREE to start work.
   A machine processes one op at a time -- no preemption.                   */
static unsigned char mach_ready[NM];


/* ---------- Best schedule found so far (the answer we return). ---------- */

/* best_makespan = the smallest total completion time discovered. Updated
   each time a complete schedule beats the current record.                  */
static unsigned char best_makespan;

/* The best schedule is stored as three parallel arrays indexed by
   "placement order" -- i.e., the order in which the search assigned the
   ops, not by (job, op). We keep the placement order because it makes the
   final printout read like an execution log of the scheduler.              */
static unsigned char best_seq_job  [TOTAL_OPS];   /* which job was placed   */
static unsigned char best_seq_op   [TOTAL_OPS];   /* which op-index of it   */
static unsigned char best_seq_start[TOTAL_OPS];   /* its start time         */

/* Current partial schedule (mirrors best_seq_* but for the in-progress
   branch). We copy cur_seq_* -> best_seq_* whenever we improve.            */
static unsigned char cur_seq_job  [TOTAL_OPS];
static unsigned char cur_seq_op   [TOTAL_OPS];
static unsigned char cur_seq_start[TOTAL_OPS];


/* ---------- Counters for the chain-of-thought trace. -------------------- */

/* nodes_explored = total recursive calls. A pure search-cost metric --
   useful for the writeup ("the solver explored N states inside the
   transformer").                                                            */
static int nodes_explored;

/* prunes = number of times we cut off a subtree using the makespan
   lower bound. Lets us show that pruning actually fires on this instance.  */
static int prunes;


/* ===========================================================================
   recurse(depth)
   ---------------------------------------------------------------------------
   Depth-first branch-and-bound over partial schedules.

   - depth: how many operations have been placed in the current branch.
            Equal to the index into cur_seq_*[] where the next placement
            will be written.

   At depth == TOTAL_OPS we've placed every op: evaluate the makespan and
   update the best schedule if improved.

   At any earlier depth, we BRANCH on which job's next op to schedule.
   We try each eligible job, recurse, then undo the placement before
   trying the next sibling.

   The PRUNE check uses a cheap lower bound on makespan -- the current
   maximum machine_ready -- because any future op can only push that
   forward. If even that LB matches/exceeds best_makespan, no completion
   of this branch can improve, so we cut.

   NOTE on attributes:
     We deliberately do NOT mark this inline. The recursion lives in the
     WASM call stack; inlining would defeat the recursive structure.
   =========================================================================== */
static void recurse(int depth) {
    nodes_explored = nodes_explored + 1;       /* count this node visit     */

    /* ----- Base case: all ops placed. ------------------------------------ */
    if (depth == TOTAL_OPS) {
        /* Makespan of a complete schedule = max machine completion time.
           Equivalent (and we could also use max job_ready) -- both are
           equal once every job's last op has been placed.                  */
        int mk = mach_ready[0];
        if (mach_ready[1] > mk) mk = mach_ready[1];
        if (mach_ready[2] > mk) mk = mach_ready[2];

        if (mk < best_makespan) {
            best_makespan = mk;
            printf("  new best makespan: %d\n", mk);

            /* Snapshot the current schedule as the new best.               */
            int i;
            for (i = 0; i < TOTAL_OPS; i = i + 1) {
                best_seq_job  [i] = cur_seq_job  [i];
                best_seq_op   [i] = cur_seq_op   [i];
                best_seq_start[i] = cur_seq_start[i];
            }
        }
        return;
    }

    /* ----- Lower-bound prune. -------------------------------------------- */
    /* LB = max machine_ready over all machines. Any future placement can
       only make some mach_ready larger, never smaller. So LB is a valid
       lower bound on the completed makespan of this branch.                */
    int lb = mach_ready[0];
    if (mach_ready[1] > lb) lb = mach_ready[1];
    if (mach_ready[2] > lb) lb = mach_ready[2];

    if (lb >= best_makespan) {
        prunes = prunes + 1;
        return;                                /* this branch can't improve */
    }

    /* ----- Branch: try each job that still has ops to schedule. --------- */
    int j;
    for (j = 0; j < NJ; j = j + 1) {
        int k = op_done[j];                    /* index of j's next op      */
        if (k >= NM) continue;                 /* job j is already done     */

        int m = route_m[j][k];                 /* required machine          */
        int d = dur    [j][k];                 /* processing time           */

        /* Earliest legal start: respects both the job's chain (predecessor
           must be done) and the machine's availability.                     */
        int start = job_ready[j];
        if (mach_ready[m] > start) start = mach_ready[m];
        int end   = start + d;

        /* ----- Apply the placement. ------------------------------------- */
        /* Save what we need to restore on backtrack. We only need to save
           the two scalars we're about to mutate (job_ready[j] and
           mach_ready[m]); op_done[j] is restored by decrement.              */
        int saved_jr = job_ready[j];
        int saved_mr = mach_ready[m];

        op_done [j]     = k + 1;
        job_ready[j]    = end;
        mach_ready[m]   = end;
        cur_seq_job  [depth] = j;
        cur_seq_op   [depth] = k;
        cur_seq_start[depth] = start;

        /* ----- Recurse into the deeper search. -------------------------- */
        recurse(depth + 1);

        /* ----- Undo (backtrack). ---------------------------------------- */
        op_done   [j] = k;
        job_ready [j] = saved_jr;
        mach_ready[m] = saved_mr;
        /* No need to clear cur_seq_*[depth] -- the next placement at this
           depth will overwrite it, and the base case only reads up to
           TOTAL_OPS - 1.                                                    */
    }
}


/* ===========================================================================
   compute(input)
   ---------------------------------------------------------------------------
   Required entry point per the transformer-vm convention. We ignore the
   input string because the instance is hard-coded (the goal of this demo
   is hand-verifiability, not generality -- a future iteration could parse
   instances from input).
   =========================================================================== */
void compute(const char *input) {
    int i;

    /* ----- Initialize all mutable state. -------------------------------- */
    /* Static arrays are zero-initialized by the WASM runtime, but we set
       them explicitly so the program is robust if it's ever called more
       than once in the same image.                                          */
    for (i = 0; i < NJ; i = i + 1) {
        op_done  [i] = 0;
        job_ready[i] = 0;
    }
    for (i = 0; i < NM; i = i + 1) {
        mach_ready[i] = 0;
    }
    best_makespan   = INF_MAKESPAN;
    nodes_explored  = 0;
    prunes          = 0;

    /* ----- Print the instance so the trace is self-contained. ----------- */
    printf("3x3 job-shop scheduling (branch-and-bound, minimize makespan)\n");
    printf("instance:\n");
    for (i = 0; i < NJ; i = i + 1) {
        printf("  job %d:", i);
        int k;
        for (k = 0; k < NM; k = k + 1) {
            printf(" (M%d,%d)", route_m[i][k], dur[i][k]);
            if (k < NM - 1) printf(" ->");
        }
        printf("\n");
    }
    printf("lower bound: %d (max machine total work)\n", 10);
    printf("\nsearching...\n");

    /* ----- Run the search from an empty schedule (depth 0). ------------- */
    recurse(0);

    /* ----- Final report. ------------------------------------------------- */
    printf("\nnodes explored: %d\n", nodes_explored);
    printf("prunes: %d\n", prunes);
    printf("optimal makespan: %d\n\n", best_makespan);

    printf("schedule (in scheduler placement order):\n");
    for (i = 0; i < TOTAL_OPS; i = i + 1) {
        int j  = best_seq_job  [i];
        int op = best_seq_op   [i];
        int m  = route_m[j][op];
        int s  = best_seq_start[i];
        int d  = dur    [j][op];
        printf("  J%d op%d on M%d: [%d, %d]\n", j, op, m, s, s + d);
    }
}


/* ===========================================================================
   REFERENCE -- for hand-verification after the run.
   ---------------------------------------------------------------------------
   Whatever optimal makespan the solver reports, you can sanity-check it by:
     1. Running this file natively (clang job_shop.c -o ... && ./...)
        and confirming the same trace, OR
     2. Manually verifying that the printed schedule is feasible (each
        job's ops appear in routing order; each machine never overlaps)
        and that no operation could be moved earlier without violating
        a precedence or capacity constraint.

   The native-compile diff is what the determinism_check + sanity diff in
   the verification step automates -- the in-transformer execution should
   produce the byte-identical output of the native binary.

   Lower bound (theoretical): 10
   Optimal makespan (computed): TO BE FILLED IN ON FIRST RUN
   =========================================================================== */
