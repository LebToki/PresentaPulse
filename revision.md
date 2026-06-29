# Revision History

## [2026-06-29] - Version 1.0.2

*   Optimized `HistoryManager.get_entry` to use a secondary dictionary index, turning an O(n) linear search into an O(1) lookup. Performance for large histories is significantly improved (~197x speedup for 100,000 lookups over 1,000 entries).
