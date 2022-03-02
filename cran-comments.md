## Test environments
* Ubuntu 18.04 LTS with gcc 10.1.0
  R version 4.1.2
* Ubuntu 18.04 LTS with gcc 10.1.0
  R devel 2022-03-01 r81837 with LTO checks
* Ubuntu 18.04 LTS with gcc 10.1.0
  R version 4.1.2 with valgrind
* Ubuntu 18.04 LTS with gcc 10.1.0
  R devel 2022-03-01 r81837 with ASAN and UBSAN
* Github actions on windows-latest (release), macOS-latest (release), 
  ubuntu-20.04 (release), ubuntu-20.04 (devel), ubuntu-20.04 (oldrelease)
* win-builder (devel, oldrelease, and release)
* `rhub::check_for_cran()`
* `rhub::check(platform = c("solaris-x86-patched", "macos-highsierra-release-cran"))`
  
## R CMD check results
There were no WARNINGs or ERRORs.

There is a NOTE about the package size in some cases.
