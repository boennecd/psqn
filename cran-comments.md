## Test environments
* aarch64-apple-darwin20 with clang version 14.0.0 and GNU Fortran (GCC) 12.2.0.
* win-builder (devel, oldrelease, and release)
  
## R CMD check results
There were no WARNINGs or ERRORs.

## Resubmission
This is a resubmission. In this version I:
* only skipped the test if OpenMP is not supported.
