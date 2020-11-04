## Test environments
* Ubuntu 18.04 LTS with gcc 8.3.0
  R version 3.6.3  
* Ubuntu 16.04 LTS (on travis-ci)
  R version 4.0.0
* Ubuntu 18.04 LTS with clang-6.0.0 with ASAN and UBSAN
  R devel (2020-11-03 r79399)
* win-builder (devel and release)
* `rhub::check_for_cran()`
* `rhub::check_on_solaris()`
* `rhub::check(platform = "macos-highsierra-release")`
  
## R CMD check results
There were no WARNINGs or ERRORs.

There is a NOTE about the package size in some cases.

I have removed the `std::defaultfloat` as this is not implemented with the 
compiler on r-oldrel-windows-ix86+x86_64.

I have attempted to fix the errors on r-patched-solaris-x86. However, 
`rhub::check_on_solaris()` still passes as it did before.
