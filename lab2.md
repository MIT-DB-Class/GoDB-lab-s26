# 6.5830/6.5831 Lab 2: Query Execution

**Assigned:** [2/24/2026]
**Due:** [3/19/2026]

## Introduction

In **Lab 2**, you will build the **Execution Engine** of GoDB. The Execution Engine consists of the core data processing
logic of a database system. GoDB uses the **Volcano Iterator Model** (also known as the Pipeline model). In this model,
a query plan is represented as a tree of operators, and each operator implements a standard interface (`Init`, `Next`,
`Close`). Query execution starts by calling `Next()` on the root operator of the tree. This operator calls
`Next()` on its children to request data, processes that data, and returns a result. This unified interface allows us
to easily implement new operators and compose them without complex orchestration logic.

## Logistics

* **Files to Modify:**
  * `godb/execution/filter_executor.go`
  * `godb/execution/projection_executor.go`
  * `godb/execution/limit_executor.go`
  * `godb/execution/seq_scan_executor.go`
  * `godb/execution/index_scan_executor.go`
  * `godb/execution/insert_executor.go`
  * `godb/execution/deletion_executor.go`
  * `godb/execution/update_executor.go`
  * `godb/execution/block_nested_loop_join_executor.go`
  * `godb/execution/aggregate_executor.go`
  * `godb/execution/sort_executor.go`
  * `godb/execution/hash_join_executor.go`
  * `godb/execution/sort_merge_join_executor.go`
  * `godb/execution/index_nested_loop_join_executor.go`
  * `godb/execution/topn_executor.go`
  * `godb/execution/materialize_executor.go`

---

## Background

### The Query Processing Pipeline
In a full database system, the user's SQL query goes through several stages:
1.  **Parser:** Converts SQL text into an Abstract Syntax Tree (AST).
2.  **Binder:** Converts the AST into a "Logical Plan" by replacing parsers symbols with concrete references to objects
    (e.g., tables, columns)
3.  **Planner/Optimizer:** Iteratively searches for optimal implementations of the logical plan, eventually producing
     a "Physical Plan" (specific algorithms, e.g., choosing Hash Join vs. Nested Loop Join). The output of the
     planner is a tree of `PlanNode`s, and it is often cached to avoid replanning for similar queries.
4.  **Executor (The focus of this lab):** Executes the physical plan. Conceptually, one could execute the physical plan
    directly, but many database systems, including GoDB, distinguish between "executors" and the physical plan. Essentially,
    executors are stateful to each query and encapsulate runtime resources, whereas plan nodes are pure and lightweight.
5.  **Storage Engine:** The Executor requests data from the `BufferPool` and `TableHeap` (You built this in Lab 1).

### GoDB's Execution Engine
GoDB's execution engine is simplified for educational purposes:
* **Iterator Model:** As mentioned, we use the `Init()` (setup), `Next()` (retrieve one tuple), and `Close()` (cleanup)
  pattern. Production systems often have more methods to pass mutable parameters, seek and rewind, etc. 
* **Eager Copy:** We eagerly copy tuples into executor buffers. This simplifies the memory management logic at the cost
  of some extra copying. Many production systems avoid this copy by passing references to storage pages and copying only
  at the end.
* **Tuple-at-a-time:** We process data one tuple at a time. This is often ok for disk-based systems, but it introduces
  multiple layers of dynamic dispatch to process one tuple, which can be much more expensive than the processing of the
  tuple itself on modern hardware. Production systems often alleviate this by processing batches of tuples at a time, and
  some may even compile a query plan to machine code (whereas the iterator model is essentially an interpreter over the
  plan) to further optimize performance.
* **Memory-Only:** In GoDB, you may assume that intermediate state (e.g., hash tables) fits in memory. This greatly 
  simplifies the implementation and allows you to focus on the core processing logic. This assumption is not true in 
  production systems, and they implement sophisticated memory management strategies to be able to process large data 
  sets that don't fit in memory.

### Tuples, Values, and Expressions
To build operators, you must first understand how data is represented in GoDB:
* **`Value`:** An interface wrapper for data types (Int, String, etc.). You will use `value.Compare(other)` and `value.Eval()` frequently.
* **`RawTuple`:** A thinly wrapped array of bytes that represents a row in the same format they exist on storage. This is the
  "tuple" format you have been working with so far. Importantly, `RawTuple` does not contain any metadata about the schema
  of the table it came from and is therefore not suitable for use in the execution engine.
* **`Tuple`:** This is the canonical representation of a row in GoDB's execution layer. Logically, it contains a list of
  `godb.Value`s and a `RecordID` (if it came from disk). Underneath the hood, like other production systems, GoDB `Tuple`
  can be backed either by a `RawTuple` along with its descriptor, an array of generated values, or a combination of both.
* **`Expr` (Expression):** The planner provides expressions (like `age > 25` or `salary * 1.1`).
    * An expression tree works similarly to an operator tree. You call `expr.Eval(tuple)` to get a result `Value`.
    * Example: A `FilterExecutor` evaluates a boolean expression. If `Eval(tuple)` returns 1, the tuple is passed up; otherwise, it is dropped.

### Indexes

Indexes are auxiliary data structures that allow for efficient retrieval of data based on the values of specific columns.
Without an index, finding specific tuples almost always requires a full sequential scan of the
table, which is prohibitively expensive for large datasets. In GoDB (and many other DBMSs), the Index is decoupled from the
actual table storage (`TableHeap`):

* **The TableHeap** stores the full content of the rows (Tuples).
* **The Index** stores a mapping from a specific Key (e.g., the value of the `id` column) to a `RecordID` (RID).

Such indexes are called **Secondary Indexes** because they are built on top of the primary table. Some systems
also support **Primary Indexes** or **Clustered Indexes**, where all tuple data is stored in an index instead of a heap.

**Your Role in this Lab:**
Building full on-disk indexes with full concurrency is a month-long project and largely orthogonal to the rest of the
database system. Instead, we have provided you with a simple indexing layer (`godb/indexing`) that is purely in-memory.
This will help you learn about how to use indexes and how they interact with the rest of the system:

1. **Read Path (Scans):** When the Query Planner chooses an Index Scan, your `IndexScanExecutor` or `IndexLookupExecutor`
   will query the index object to retrieve a list of RIDs. You must then use these RIDs to fetch the actual full Tuples 
   from the `TableHeap`.
2. **Write Path (Maintenance):** This is the most critical part of the modification operators (`INSERT`, `UPDATE`, `DELETE`).
   The database system must guarantee that the Table and the Index are always consistent.
* **Insert:** After inserting a tuple into the `TableHeap` and obtaining a new RID, you must insert entries into
  **every** index defined on that table.
* **Delete:** You must remove the entry corresponding to the deleted tuple's RID from **every** index.
* **Update:** If the value of an indexed column changes, you must update the index. This is effectively a Delete of the
  old key followed by an Insert of the new key.

  
### Memory Management

Efficient memory usage is essential for performance in database systems. Although GoDB does not need to handle
larger-than-memory datasets, you must still be frugal with memory usage to avoid running out of memory. One of the key
benefits of the iterator model is that a tuple only needs to be valid until the `Next()` call. Therefore, executors
typically allocate memory for only one tuple, and repeatedly populates that buffer with new data, passing up a reference
to it until the query is complete. This leads to the key considerations of **Tuple Stability**. Because of memory reuse,
it is **not** safe to store a tuple in a data structure that will outlive the `Next()` call. If an executor needs to
store a tuple for  longer (e.g., storing a tuple in a list for a `BlockNestedLoopJoin`, or saving it in a hash table
for `Aggregation`), you **must** copy of that tuple into stable memory. The one exception is indexes: our implementations
automatically copy tuples when needed, so you do not need to worry about copying when passing it into an index.

In addition to aggressive memory reuse, production systems additionally optimize memory allocation when allocation
is needed (e.g., when sorting a large number of tuples). Common approaches include arena allocation, which allocates
large blocks of contiguous memory at startup that are only freed once the query is complete, and pooling, which caches
and reuses allocated objects instead of freeing them immediately. You are **not** required to implement these
strategies in this lab, but know that they are essential for performance in a production system.

---

## Part 1: Warm-Up (Basic Operators)
We start with the simplest operators. 

* **Sequential Scan**: Iterate over a TableHeap using the Iterator you implemented in Lab 1.
This operator is the leaf node of many query plans and acts as the source of data. Ensure you handle the Iterator
correctly in Next() and Init() (resetting the iterator).
* **Filter**: Iterate over the child operator. For each tuple, evaluate the predicate (expr.Eval). If the result is true,
return the tuple. If false, fetch the next one from the child.
* **Projection**: Iterate over the child. For each tuple, calculate a new set of values by evaluating the list of
expressions provided in the plan. Create a new tuple with these values and return it.
* **Limit**: Keep a count of how many tuples you have emitted. Once you reach the limit, return false on the Next() call
to stop the pipeline, even if the child has more data.

**Test:**
Run `go test -v ./execution -run BasicExecutor`

---

## Part 2: Access Methods & Modifications

These operators interface directly with the storage engine (Lab 1) and the Indexing layer. For simplicity, we assume
that all scans return the entire tuple. In many production systems, each executor can be configured to only select the
necessary columns to minimize memory usage, but doing so is quite tedious for educational purposes.

### 1. Index Scans
* **Index Scan (`IndexScanExecutor`)**: Iterate over a B+Tree index. Use the provided `indexing.Index` interface to
  perform a range scan (`Scan`) starting from a specific key.
* **Index Lookup (`IndexLookupExecutor`)**: Perform a point lookup. Use `index.ScanKey` to retrieve RecordIDs (RIDs) for
  a specific key, then fetch the actual tuples from the `TableHeap` using those RIDs.

**Test:**
Run `go test -v ./execution -run IndexExecutor` (NOTE: run this test only after you finish Modifications)

### 2. Modifications
**Crucially, you must maintain index consistency.**
* **Insert (`InsertExecutor`)**: Insert tuples into the `TableHeap`. You must also insert the corresponding key into *all*
  active indexes defined on the table.
* **Delete (`DeletionExecutor`)**: Delete tuples from the `TableHeap`. You can assume that the child will read the entire
  tuple. You must also remove the corresponding keys from all active indexes.
* **Update (`UpdateExecutor`)**: Update the tuple in the `TableHeap`. Check if the updated columns are part
  of any index key. If so, update the index entries (delete old key, insert new key).

NOTE: Insert, Delete, Update executors follow standard SQL semantics. These executors should consume all tuples from their child executors and return a single output containing the number of rows affected (e.g. rows inserted, deleted, or updated).

**Test:**
Run `go test -v ./execution -run ModificationExecutor`



---

## Part 3: Core Processing Operators

These operators are more complex and may require managing memory buffers or hash tables. You may use `execution/hash_table.go`
file as a starting point for your implementation.

### Background: Handling NULLs

One of the most notoriously tricky aspects of building a SQL-compliant engine is handling `NULL` values correctly.
`NULL` in SQL specifically means **"Unknown"**, and it has major implications for your implementation:

#### Three-Valued Logic

Boolean expressions in SQL do not just evaluate to `TRUE` or `FALSE`; they can also evaluate to `NULL` (Unknown).

* **Propagation:** Arithmetic or comparison with a `NULL` almost always results in `NULL` (e.g., `5 + NULL = NULL`,
  `5 > NULL = NULL`).
* **Filtering:** The `WHERE` clause only accepts tuples where the predicate evaluates to **strictly `TRUE`. It
  discards tuples where the result is `FALSE` **or** `NULL`.
* **Implementation:** In `godb/planner/expr.go`, you will see helper functions like `ExprIsTrue`. When evaluating
  predicates in your `FilterExecutor` or `JoinExecutor`, make sure you use these helpers to correctly treat `NULL`
  results as "reject."

#### Equality Logic

As you might have realized by now, three-valued logic is very strange because `NULL = NULL` is **NULL**, which is not
true nor false. Two unknown values are not necessarily equal or not equal. Consequently, there are a few things that you
must keep in mind when implementing equality:
* **Physical Equality:** Enforcing three-valued logic in the storage and indexing layer would obviously break any data
  structure that uses equality or comparisons. Therefore, GoDB does not handle `NULL`s at that level. In GoDB, `NULL`
  values are represented as special byte patterns (0xFF for strings, and int_min for ints)., and comparisons are done
  normally (these methods are provided for you in `godb/types.go` and `godb/indexing/key.go`)
* **Expressions:** The expression evaluator strictly enforces three-valued logic. This means that expressions like
  `age > 25` will evaluate to `NULL` if the `age` column is `NULL`, and importantly, `WHERE age = NULL` will also evaluate to
  `NULL`. There is a special syntax for testing for `NULL` values: `age IS NULL` and `age IS NOT NULL`.
* **Joins:** Keys with `NULL`s do **NOT** join with each other as their equality is unknown. You may need to implement
  special logic to handle this case.
* **Aggregation:** As a general rule, aggregates treat `NULL` values as equal when grouping, but `NULL`s
   are ignored in the actual aggregation (e.g., `COUNT(x)` counts only non-null `x`). If a group contains only `NULL`s,
   the aggregate returns `0` for `COUNT`, but `NULL` for other types of aggregates.
* **Ordering:** Ordering by a column with `NULL` should again treat `NULL` as equal. You may use the `Compare()`
  method from `godb/types.go` to order. Production systems may additionally allow users to specify an ordering mode
  (e.g., NULLS FIRST or NULLS LAST).

### 1. Block Nested Loop Join (`BlockNestedLoopJoinExecutor`)
Standard Nested Loop Join is slow because it scans the inner table once for every single row of the outer table. You may assume
that the output tuple is the concatenation of the two input tuples from left to right.
* **Optimization:** Load a "block" (chunk) of tuples from the outer (left) child into memory.
* **Execution:** Scan the inner (right) child *once* for that entire block. Compare every tuple in the inner scan against
  every tuple in the memory block.
* This reduces the number of times we scan the inner table by a factor of `BlockSize`.

**Test:**
Run `go test -v ./execution -run BNLJ`

### 2. Sort (`SortExecutor`)
* **Execution:** The first call to `Next()` must consume **all** tuples from the child, store them in a buffer, and sort
them according to the `OrderBy` expression. Subsequent calls simply returns the next tuple from the sorted buffer.
* **Note:** This is a "blocking" operator; it produces no results until the child is exhausted.
  
**Test:**
  Run `go test -v ./execution -run Sort`

### 3. Aggregation (`AggregateExecutor`)
* **Logic:** Implement grouped aggregation (e.g., `SELECT count(*), sum(a) FROM table GROUP BY b`).
* **Strategy:** Use **Hash Aggregation**.
    * In the first call to `Next()`, consume *all* tuples from the child.
    * Maintain a Hash Map (you may use the one in `execution/hash_table.go`) where keys are the `GROUP BY` fields and
      values are the running aggregates (Sum, Count, Min, Max).
    * Update the aggregate state for every tuple.
    * Subsequent calls to `Next()` iterate through the populated hash map and return the results.

**Test:**
Run `go test -v ./execution -run Aggregate`

---

## Part 4: Advanced Operators (Pick 2)

You must implement **two** of the following advanced operators. Choose the ones that interest you most.

### Option A: Hash Join (`hash_join_executor.go`)
Implement the classic Hash Join, usually faster than Nested Loop for equality joins.
* **Build Phase:** Consume all tuples from the left child and build an in-memory hash table keyed by the join attribute.
* **Probe Phase:** Iterate over the right child. For each tuple, hash its join key and look up matches in the table.

**Test:**
Run `go test -v ./execution -run HashJoin`

### Option B: Sort-Merge Join (`sort_merge_join_executor.go`)
Implement Sort-Merge Join.
* **Assumption:** The planner guarantees that children are already sorted on the join key (or you must inject a sort).
* **Execution:** Iterate through both children simultaneously using a "zipper" approach to find matches. This is very efficient for sorted data.

**Test:**
Run `go test -v ./execution -run SortMergeJoin`

### Option C: Index Nested Loop Join (`index_nested_loop_join_executor.go`)
An optimization of Join where the right table has an index on the join key.
* **Execution:** Iterate over the left child. For each tuple, do not scan the right table. Instead, use the right table's Index to look up matching RIDs, then fetch the specific tuples.

**Test:**
Run `go test -v ./execution -run IndexNestedLoopJoin`

### Option D: Top-N Optimization (`topn_executor.go`)
Implement `TopNExecutor`.
* **Goal:** Efficiently handle `ORDER BY ... LIMIT N`.
* **Optimization:** Instead of sorting the *entire* table (O(N log N)) and taking the top K, use a **Min-Heap** of size K.
* **Logic:** Pass through the data once. Maintain the heap such that it always contains the "best" K tuples seen so far.

**Test:**
Run `go test -v ./execution -run TopN`

### Option E: Materialization (`materialize_executor.go`)
Implement `MaterializeExecutor`.
* **Use Case:** Sometimes the same subtree needs to be scanned multiple times (e.g., the inner table of a Nested Loop Join).
* **Logic:** On the first pass, consume the child and store all tuples in an in-memory buffer. On subsequent passes, replay
  the tuples from the buffer instead of re-executing the child tree.

**Test:**
Run `go test -v ./execution -run Materialize`

---
## Part 5: Putting It All Together (Experimental)

While most of this lab involves passing unit tests, we have provided an experimental **godb command-line tool** in `godb/main.go`
that allows you to play with an end-to-end query processing pipeline. This tool includes a parser, binder, optimizer, and your
execution engine.

### 1. Enabling Physical Optimizer Rules
Before running the shell, you must enable the physical optimizer rules that correspond to the operators you implemented in
**Part 4**. By default, these rules are commented out in the source code, so the command line tool will not use them.
Open `godb/main.go` and locate the `physicalRules` slice inside the `NewGoDB` function. Uncomment the rules you have implemented:

### 2. Using GoDB

To test the shell, you must first load data using a catalog and CSV files. We have provided sample
[TPC-C](https://www.tpc.org/tpc_documents_current_versions/pdf/tpc-c_v5.11.0.pdf) data in the `example_data` directory.
First, batch load the data into the database:

```bash
go run main.go load -catalog example_data/example-catalog.json [-truncate] example_data/warehouse.csv example_data/district.csv example_data/customer.csv example_data/orders.csv
```

Once the data is loaded, you can start the interactive SQL shell. We recommend using the `-explain` flag to see the physical plan chosen by the optimizer before it is executed:

```bash
go run godb/main.go shell -catalog example_data/example-catalog.json -explain
```

Try running queries that trigger the operators you have implemented. For example:

**Filter and Aggregate:**
```sql
SELECT c_credit, COUNT(*) FROM customer GROUP BY c_credit;
```

**Join:**
```sql
SELECT c_first, c_last, o_id 
FROM customer, orders 
WHERE c_id = o_c_id AND c_w_id = o_w_id AND c_d_id = o_d_id 
AND c_last = 'Smith';
```

The `explain` output will show you a tree visualization of the plan, allowing you to verify if the optimizer correctly chose your
advanced operators (e.g., `HashJoin` vs. `BlockNestedLoopJoin`) based on the query predicates and available indexes.

### 3. Have Fun!
This task is not graded, you should simply have fun with your personal DBMS! You can supply GoDB with a new schema, load new data,
and play with different queries. If you find bugs in the command line tool, or feel like implementing a new feature from scratch
(e.g., DISTINCT), you may email us to discuss extra credit.

---
## Grading and Submission

### 1. Submission

This lab has an autograded component. Create a zip file containing your `godb` directory and your write-up.

```bash
zip -r lab2_submission.zip . -x "*.git*"


```
Upload this zip file to [Gradescope].

We reserve the right to re-execute tests after the deadline, as concurrency bugs are often non-deterministic. We also
reserve the right to run additional hidden tests on your code. It is your responsibility to ensure that your code can
reliably pass all tests under repeated runs and different system conditions.

### 2. Lab Write-up

You should expect to complete a short write-up in-class about the lab. To get full credit, you should be prepared to answer the following:

* Basic conceptual questions about the codebase and the programming task.
* Questions about your design decisions (e.g., locking strategy in BufferPool).
* Any challenges you faced, the amount of time you spent on the lab, and feedback on the lab for future semesters.

**Grading Breakdown:**

* **60%**: Passing public unit tests.
* **40%**: Manual grading of code quality, hidden tests, and write-up.

Good luck!