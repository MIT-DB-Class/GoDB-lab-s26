# 6.5830/6.5831 Lab 1: Storage & Buffer Management

**Assigned:** 02/03/2026
**Due:** 02/26/2026

## Introduction

In the lab assignments for 6.5830/6.5831, you will build a basic disk-based database management system called **GoDB**.
As the name suggests, GoDB is written in Go. Go is a simple, modern language that is efficient and easy to learn. If
you are new to Go, we recommend the [Tour of Go](https://go.dev/tour) to get started.

In this lab, you will build the foundation of GoDB: the **Storage Engine**. Traditional database system storage revolves
around disk pages; the engine moves data between disk and memory, organizes that data within fixed-size pages, and
provides an interface for higher-level operators to read and write tuples.

Your task is to implement the storage layer from the bottom up. You will start with low-level bit manipulation,
move to organizing data within a memory page, implement a thread-safe Buffer Pool Manager to cache these pages, and
finally implement `TableHeap` to manage a collection of pages as a table.

## Logistics

**Files to Modify:**
* `godb/storage/bitmap.go`
* `godb/storage/heap_page.go`
* `godb/storage/page.go`
* `godb/storage/buffer_pool.go`
* `godb/execution/table_heap.go`

---

## Part 0: Warm-up - Bitmaps
**File:** `godb/storage/bitmap.go`

To efficiently track allocated slots in a page or free pages in a file, GoDB uses bitmaps. A bitmap is a compact array
of bits where the i-th bit corresponds to the i-th resource (e.g., the i-th slot on a page). Bitmaps are widely used in
database systems because they provide exceptional space efficiency compared to alternatives such as a list of integers.
They also enable high-performance scans by checking 64 bits at a time using standard CPU word operations, the system can
quickly identify free resources while keeping the metadata structure small enough to remain CPU-cache resident.

Another common trick in high-performance database systems is the use of **Blittable Types**. In standard applications, you
might use objects or structs with pointers (e.g., a linked list or a JSON tree). However, writing these to disk requires
CPU-intensive **serialization** (converting pointers and structures into a flat byte stream). In contrast, blittable
types are organized in memory exactly as they should appear on disk. This allows the system to copy bytes from disk
directly when reading and "cast" a pointer without complex decoding steps and vice versa for writes. This "Zero-Copy"
and "Zero-Allocation" philosophy is crucial for performance, and you will see it used in many places throughout GoDB,
including the bitmap.

**A Note on Alignment:** You will notice that GoDB ensures data is aligned to 8-byte boundaries (e.g., godb.AlignedTo8).
This is critical when casting raw byte slices to types like uint64. On some modern architectures (like ARM), accessing a
multi-byte integer at an unaligned address causes a hardware crash (bus error). On x86, it works but is significantly
slower, and it will silently fail on some guarantees (e.g., atomically writing a misaligned memory location may **not**
actually be atomic). It is therefore standard practice to ensure all data structures are aligned in modern systems.
Most of the time, the go compiler will automatically align your structs for you. However, when handcrafting blittable
structs, you often must manually ensure alignment. One notable exception to this rule in GoDB is the Write-Ahead Log.
Logs are often packed as tightly as possible to minimize disk I/O, sacrificing some CPU cycles for storage efficiency.
Because logs are seldomly read (unless during recovery), and almost never read in a tight loop, the performance trade-off
is acceptable.

### Implementation Tasks
You need to implement the following methods in `Bitmap`:

1. **`SetBit(i int, on bool)`**: Set the bit at index `i` to 1 (if `on` is true) or 0 (if `on` is false). This method
   should return the *previous* value of that bit.
2. **`LoadBit(i int)`**: Return true if the bit at index `i` is set to 1, false otherwise.
3. **`FindFirstZero(startHint int)`**: Find the index of the first bit set to 0, starting the search from `startHint`.
   If the end of the bitmap is reached, wrap around to 0. Return -1 if all bits are set to 1.

**Implementation Hint:** The blittable struct definition of Bitmap and the way to construct it from raw bytes is provided
for you. The underlying data is stored as a slice of `uint64`. Use standard bitwise operators (`&`, `|`, `<<`) to access
bits, and think about how you can optimize the scan in `FindFirstZero`.

**Test:**
Run `go test -v ./storage -run Bitmap`

---

## Part 1: Page Layout (HeapPage)

**File:** `godb/storage/heap_page.go`

Database tables are stored in fixed-size blocks of memory called Pages. A traditional Slotted Page used by production
systems is designed to handle variable-Length records (e.g., strings of different lengths). It typically places a header 
at the beginning of the page and an array of slot pointers at the *end* of the page growing inward, with data growing
outward. In GoDB, storage is simplified because we only support **Fixed-Length Tuples**. A column in GoDB is either an
8-byte integer, or a 32-byte (padded) string. This allows us to pre-calculate the location of slots and directly access
them using simple arithmetic rather than storing an array of offsets.

GoDB uses a page size of **4KB** (`godb.PageSize`). We chose 4KB because most modern filesystems (ext4, NTFS) and SSDs
are able to **atomically** write blocks of 4KB. This means that if a system crashes during a write, either the entire
page will be lost or none of it. In production systems, page sizes are often larger (e.g., 16KB), and they implement
additional logic to ensure data integrity (e.g., double-writing), adding considerable complexity that is beyond the
scope of this lab. For GoDB, you can assume that page writes are atomic.

**Page Layout & Alignment**
The specific layout of a `HeapPage` is:
`[ Header | Allocation Bitmap | Deleted Bitmap | ... Tuples ... ]`

Because GoDB optimizes for performance by casting raw byte slices directly into structs or `uint64` arrays (for bitmaps),
**memory alignment** is critical. You must ensure that the Bitmaps and the Tuple Data start at 8-byte boundaries. The
header includes padding to ensure this alignment is maintained.

### The Allocation vs. Deletion Distinction

You will notice the page layout contains **two** bitmaps:

1. **Allocation Bitmap:** Indicates if a slot has valid data (is "occupied").
2. **Deleted Bitmap:** Indicates if the data in an occupied slot has been logically deleted.

Why do we need both? In a database supporting transactions (which you will implement later in the semester), deleting a
record doesn't remove it immediately. If the transaction deleting the record fails to commit, we will need to restore
the data, and physical deletion will greatly complicate the process because another tuple may have been inserted in its
place. Most systems would therefore separate the logical deletion from physical deletion, and implement a separate
garbage-collection pass to reclaim space.

### Implementation Tasks

You must implement the following in `HeapPage`:

1. **Header Getters/Setters**: Implement `NumUsed`, `setNumUsed`, `NumSlots`, and `RowSize`. These methods read/write fields within the
   header in the `Bytes` array.
2. **`InitializeHeapPage`**: Initialize a new heap page. You must compute the page layout and allow space for the header, the two bitmaps,
   and the tuples themselves. The bitmaps must be padded to 8-byte alignment.
3. **`AsHeapPage`**: This function "casts" a raw `PageFrame` into a `HeapPage`. You may need to compute and cache some frequently accessed
   offsets for performance.
4. **`MarkAllocated` / `MarkDeleted` / `IsAllocated` / `IsDeleted`**: Wrapper methods that use your Bitmap implementation to track the state
   of specific slots. Note that `MarkAllocated` should also update the `NumUsed` counter in the header.
5. **`AccessTuple`**: Return a `RawTuple` (byte slice) pointing to the data for a specific RecordID.

**A Note on LSNs:**
You will notice the first field in the page header is an **LSN** (Log Sequence Number). This is used to track the order
of operations for Crash Recovery (Write-Ahead Logging). You will implement recovery in a future lab. For now, you simply
need to ensure your layout calculations account for the 8 bytes reserved for the LSN at the start of the page; you do not
need to read or write this value manually in this lab.

**Test:**
Run `go test -v ./storage -run HeapPage`

---

## Part 2: Buffer Pool Manager

**File:** `godb/storage/buffer_pool.go`

The Buffer Pool is the heart of the database system. It caches pages in memory to allow access and minimize expensive
disk I/O. Because the database engine is highly concurrent, your Buffer Pool must be **thread-safe** and efficient.
You are allowed to use the Go standard  library along with provided third-party ones like `xsync.MapOf` to implement
the page table.

### GoDB Disk Abstraction

The Buffer Pool does not access the file system directly. Instead, it relies on two interfaces defined in `godb/storage/db_file.go`:

* **`DBFileManager`**: Manages a collection of files, usually mapping table names or ObjectIDs to physical files.
* **`DBFile`**: Represents a specific file on disk. It handles the low-level `ReadPage` and `WritePage` operations.

When `GetPage` is called, the Buffer Pool checks its cache. If the page is missing, it asks the `DBFileManager` for the
appropriate `DBFile` and reads the page from disk. Basic implementations for these interfaces are provided in the skeleton code.
Such an abstraction allows us to easily swap out disk implementations for testing or mocking, and also allows us to supply
alternative implementations for production systems (e.g., a cloud-native system may use object storage instead of local disks).

### Pinning and Dirty Pages

To ensure data integrity and memory safety, the Buffer Pool needs to track for every page:

1. **Pin Count:** When an operator (like a Scan or Insert) needs a page, it asks the Buffer Pool to **pin** it. A pinned page
   is "in use" and **cannot be evicted**. If the Buffer Pool evicted a page while an operator was reading a memory pointer inside it,
   the operator would read garbage data (or crash). Operators must call `UnpinPage` when they are finished.
2. **Dirty Bit:** If an operator modifies a page (e.g., updates a tuple), it must indicate this by setting `setDirty=true`
   when unpinning. **Dirty pages** contain data that has not yet been saved to disk. If the Buffer Pool decides to evict a dirty 
   page, it *must* first write that page back to disk to ensure the changes are not lost.

### Performance Requirements

When synchronizing access to the buffer pool, you must avoid coarse-grained latches that serialize disk reads and writes
(see `TestBufferPool_IOConcurrency`). For the eviction policy you choose, you must ensure that it is resilient to
"scan pollution". Scan pollution occurs when the database system accesses many pages as part of a sequential scan, and 
as a consequence, evicts real hot pages (e.g., root node of a tree index) in exchange for cold data. A naive LRU policy
would cause scan pollution because all scanned pages are promoted to the front of the recency queue. Take a look at the
passing criteria in `TestBufferPool_Concurrent_ScanResistance` for more information.

### Implementation Tasks

You need to implement the following methods in `BufferPool`:

1. **`NewBufferPool`**: Initialize your structures. Systems often preallocate all `PageFrame`s at startup to avoid allocation overhead during execution.
2. **`GetPage(pageID)`**: This is the most complex method. It must handle two distinct paths:
* **Cache Hit:** The page is in the `pageTable`. You must pin it and return it. *Concurrency warning:* Ensure the page isn't being evicted while you are trying to pin it.
* **Cache Miss:** You will need to find a victim (writing it to disk if it is dirty), and then read the requested data into the now free frame.
3. **`UnpinPage`**: Decrement the pin count. If `setDirty` is true, mark the page as dirty.
4. **`FlushAllPages`**: Scan the frame array and write all dirty pages to disk, ignoring pins.
* **Why ignore pins?** Unlike eviction, flushing does not remove the page from memory; it simply synchronizes the disk with memory.
    Therefore, it is safe to flush a page even if it is currently pinned by another thread. However, you must still physically protect the bytes
    within that page from concurrent modification.
* **Future Use:** While used primarily for testing in Lab 1, this method is the foundation for **Checkpointing**, which you will implement
    in the Recovery Lab. You will see a `flushedUntil` LSN parameter. For this lab, you can ignore it.

**A Note on Debugging**: Concurrent programming is difficult as bugs are often non-deterministic. You may
encounter "Heisenbugs"—bugs that disappear when you add `fmt.Println` statements or run the debugger, but reappear in production.
There is no single solution to this problem, and you will need to experiment with different approaches to debug your code. First,
you should try to find ways to reproduce the bug and tune the test case to increase the likelihood of reproducing it. Often, you will
need to reason hard about possible interleavings and establish invariants to help you understand the behavior of your code. Some tools
that can help you debug concurrency bugs include:
* **Use the Race Detector:** Run your tests with `go test -race ...`. This is the most effective way to catch data races (e.g., two threads modifying `pinCount` without atomic operations or locks).
* **Run Loop:** If a test fails "sometimes," run it in a loop: `go test -count=100 ...`.

**Test:**
Run `go test -v ./storage -run BufferPool`

---

## Part 3: Table Heap & Iterators

**File:** `godb/execution/table_heap.go`

The `TableHeap` represents a physical table on disk, which is a collection of `HeapPage`s. It is responsible for
providing the low-level interface that allows the database execution engine to perform tuple-level operations—such as
insertions, updates, deletions, and scans, while abstracting away the complexity of physical page tracking and memory
management. This layer is often referred to as **Access Methods** in production systems. 

While it may sound like `TableHeap` is just a thin wrapper around `HeapPage`s, it is responsible for space 
management, which can be quite complex. In a production DBMS, insertion performance is critical. Scanning the
file to find a free slot (First-Fit) is too slow because it requires reading many pages from disk, and can be linear to
the number of pages in the file. Conversely, always appending to the end (Append-Only) is fast but wastes space if
records are deleted. Production systems often need to devise additional mechanisms to track space usage within the 
heap file and balance between insertion performance and space utilization. For most databases, however, the assumption
is that deletion is a rare operation and is often performed as background maintenance (e.g., nightly). Therefore, for
this lab, you will focus on the append-only strategy and may attempt to implement a more sophisticated strategy to
reuse slots for extra credit.

Your `TableHeap` implementation should be thread-safe by making use of page-level latches. You will notice many methods
(like `ReadTuple` or `Iterator`) take a `[]byte` buffer as an argument. In high-performance systems, allocating new memory
(e.g., `make([]byte)`) for every single tuple read (i.e., in a tight loop) creates massive pressure on the memory allocator
and garbage collector (of higher-level languages like Go), which will tank performance and scalability. Therefore, you
will see many methods passing a reusable `buffer` parameter that can be reused to avoid unnecessary allocations. This
will be especially important in later labs. 

**A Note on Transactions and Logging:**
You will notice concepts like `TransactionContext` and `LogManager` in the code base. **For Lab 1, you do not need to
implement transactions or logging.** You can ignore these parameters for now.

### Implementation Tasks

You need to implement the following:

1. **`InsertTuple`**: Find a slot in the table and insert the tuple, allocating new pages if necessary.
2. **`DeleteTuple` / `ReadTuple` / `UpdateTuple`**: Fetch the page containing the RecordID from the Buffer Pool and perform the requested operation.
3. **`Iterator`**: Implement the `TableHeapIterator` struct, and associated methods. This should allow you to scan the entire 
   table in a single pass. When scanning, you must skip allocation map pages, empty slots, and deleted tuples.
4. **`VacuumPage`**: Finally, implement this method to reclaim space in a specific page (useful for background garbage collection), and update the 
   associated allocation map.  You do not need to implement the logic to invoke this method in this lab.

**Test:**
Run `go test -v ./execution -run TableHeap`

---

## Part 4: Slot Reuse (Extra Credit)

In the standard implementation of `InsertTuple`, you focused on the "Append-Only" strategy (always adding to the last
page). As mentioned, this approach maximizes **Insertion Performance** at the cost **Space Utilization**.

For extra credit, your task is to modify `InsertTuple` to efficiently find and reuse free slots within the table, without
incurring significant overhead. This is an open-ended design challenge that explores the fundamental trade-off
of **Insert Latency vs. Space Utilization**. You will likely need to add metadata pages to the heap to track space usage
within the heap. We will provide a variety of traces with different access patterns, and we will measure your
implementation's average insertion latency and space utilization over a series of runs. There is no single "correct"
algorithm here. You are free to design your own strategy and justify your solution. If you attempt the extra credit,
you should be prepared to explain your design choices and justify your tradeoffs.

---
## Grading and Submission

### 1. Submission

This lab has an autograded component. Create a zip file containing your `godb` directory and your write-up.

```bash
zip -r lab1_submission.zip . -x "*.git*"

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

* **60%**: Passing public unit tests (Bitmap, HeapPage, BufferPool, TableHeap).
* **40%**: Manual grading of code quality, hidden tests, and write-up.

Good luck!
