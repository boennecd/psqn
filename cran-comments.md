## Test environments
* Ubuntu 18.04 LTS with gcc 8.3.0
  R version 3.6.3
* Ubuntu 18.04 LTS with gcc 8.3.0
  R version 3.6.3 with `--use-valgrind`  
* Ubuntu 16.04 LTS (on travis-ci)
  R version 4.0.0
* Ubuntu 18.04 LTS with clang 6.0.0 with ASAN and 
  UBSAN checks
  R devel (2020-10-17 r79346)
* win-builder (devel and release)
* `rhub::check_for_cran()`
  
## R CMD check results
There were no WARNINGs.

There is a NOTE about the package size in some cases.

There is a note about a possibly mis-spelled word in the DESCRIPTION. 
However, Nocedal is correct.

There is an ERROR only with `rhub::check_with_sanitizers()`. The error is: 
> attributes.cpp:169:11: runtime error: load of value 2, which is not a valid value for type 'bool'
>    #0 0x7f80f0fec7ca in Rcpp::attributes::Type::Type(Rcpp::attributes::Type const&) /tmp/RtmpXwsj13/R.INSTALLb7607bdba3/Rcpp/src/attributes.cpp:169

However, this seems like the issue described here: 
http://lists.r-forge.r-project.org/pipermail/rcpp-devel/2019-February/010296.html

and it is unrelated to this package it seems (see 
https://github.com/RcppCore/Rcpp/issues/1112).
