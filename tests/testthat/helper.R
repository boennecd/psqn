# skips test on macOS
skip_on_macOS <- function()
  skip_if(Sys.info()["sysname"] == "Darwin")

# the temporary directory we will use. This must be deleted!
.temp_dir_to_use <- file.path(tempdir(), "tmp-psqn-cpp-dir")
if(!dir.exists(.temp_dir_to_use))
  dir.create(.temp_dir_to_use)

# compiles a C++ file from the inst directory and returns the needed information
# to delete all associated files and reset the working directory
compile_cpp_file <- function(f, new_name = f, do_compile = TRUE){
  old_wd <- getwd()
  # create the directory we will use
  success <- FALSE
  new_wd <- .temp_dir_to_use
  on.exit({
    if(!success)
      setwd(old_wd)
  }, add = TRUE)
  # copy the file
  file.copy(system.file(f, package = "psqn"),
            file.path(new_wd, new_name))
  # we have to copy all the headers
  file.copy(
    list.files(system.file("include", package = "psqn"), full.names = TRUE),
    new_wd)
  # set the working directory to the directory with all the files
  eval(bquote(setwd(.(new_wd))))
  if(do_compile)
    Rcpp::sourceCpp(new_name)
  success <- TRUE
  list(old_wd = old_wd, new_wd = new_wd)
}

# resets everything following a call to compile_cpp_file
reset_compile_cpp_file <- function(reset_info)
  setwd(reset_info$old_wd)

has_openmp <- \(){
  make_conf_file <- list.files(
    path=file.path(R.home(), "etc"), pattern="Makeconf$", full.names = TRUE, recursive=TRUE
  )
  if(length(make_conf_file) < 1)
    return(FALSE)

  make_conf <- readLines(make_conf_file[1])
  to_find <- "SHLIB_OPENMP_CXXFLAGS = "
  for(conf_line in make_conf){
    if(startsWith(conf_line, to_find))
      return(nchar(conf_line) > nchar(to_find))
  }
  return(FALSE)
}
