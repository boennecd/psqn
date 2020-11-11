## Test environments
* Ubuntu 18.04 LTS with gcc 8.3.0
  R version 3.6.3  
* Ubuntu 18.04 LTS with gcc 8.3.0
  R version 3.6.3 with LTO checks
* Ubuntu 16.04 LTS (on travis-ci)
  R version 4.0.0
* win-builder (devel and release)
  
## R CMD check results
There were no WARNINGs or ERRORs.

There is a NOTE about the package size in some cases.

The only error that was left is the issue with LTO. However, notice that 
this is because of an update of testthat on 2020-10-31. In particular, they 
changed this line: 
https://github.com/r-lib/testthat/blob/b684f1ee7fd806ff55a9d901174c4cc77c38b7d0/inst/include/testthat/testthat.h#L165

I submitted my package on 2020-11-04 and did not think that there would be 
breaking changes in the testthat package.

For the record, the LTO did __not__ remain. I submitted version 0.1.1 prior 
to the testthat 3.0.0 release with the breaking changes. I suspect that a 
lot of packages will now fail the LTO check. See 
https://github.com/r-lib/testthat/issues/1230
