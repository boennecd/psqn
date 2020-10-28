## Test environments
* Ubuntu 18.04 LTS with gcc 8.3.0
  R version 3.6.3  
* Ubuntu 16.04 LTS (on travis-ci)
  R version 4.0.0
* win-builder (devel and release)
* `rhub::check_for_cran()`
* `rhub::check_on_solaris()`
* `rhub::check(platform = "macos-highsierra-release")`
  
## R CMD check results
There were no WARNINGs or ERRORs.

There is a NOTE about the package size in some cases.

There is a note about a possibly mis-spelled word in the DESCRIPTION. 
However, Nocedal is correct.

I have added the missing headers which caused a build error on Solaris. 

Regarding the rchk message. then I have removed the calls to `Rf_asReal`. 
For the record, I already checked that 
`Rf_isReal(x) and Rf_isVector(x) and Rf_xlength(x) == 1L` before calling 
`Rf_asReal`. 

The test error on macOS has been fixed. There is now an example that does 
and does not use OpenMP.
