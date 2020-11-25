## Test environments
* Ubuntu 18.04 LTS with gcc 8.3.0
  R version 3.6.3  
* Ubuntu 18.04 LTS with gcc 8.3.0
  R version 3.6.3 with LTO checks
* Ubuntu 18.04 LTS with gcc 8.3.0
  R version 3.6.3 with valgrind
* Ubuntu 18.04 LTS with clang-6.0.0 
  R version 3.6.3 with ASAN and UBSAN
* Ubuntu 16.04 LTS (on travis-ci)
  R version 4.0.0
* win-builder (devel and release)
* `rhub::check_for_cran()`
* `rhub::check_on_solaris()`
* `rhub::check(platform = "macos-highsierra-release")`
* `rhub::check_with_valgrind()`
  
## R CMD check results
There were no WARNINGs or ERRORs.

There is a NOTE about the package size in some cases.
