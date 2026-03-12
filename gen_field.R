library(spmodel)

# Modified from:
# https://stackoverflow.com/a/78162082
get_current_file_path <- function() {
  this_file <- grep("^--file=", commandArgs(), value = TRUE)
  this_file <- gsub("^--file=", "", this_file)
  if (length(this_file) == 0) this_file <- rstudioapi::getSourceEditorContext()$path
  if (this_file == "") warning("Most likely running code in RStudio in a new R script window.")
  return(dirname(this_file))
}

exponential_spcov_params <- function(current_phi) {
  spcov_params(
    "exponential",
    de = 1,
    ie = 0,
    range = current_phi
  )
}

wave_spcov_params <- function() {
  spcov_params(
    "wave",
    de = 1,
    ie = 0,
    range = 2
  )
}

spherical_spcov_params <- function(grid_n) {
  spcov_params(
    "spherical",
    de = 1,
    ie = 0,
    range = floor(grid_n * 0.9)
  )
}

args <- commandArgs(trailingOnly = TRUE)

grid_n <- as.numeric(args[1])
current_phi <- as.numeric(args[6]) * grid_n
current_seed <- as.numeric(args[3])

save_path <- paste0(get_current_file_path(), "/", args[2])

x_coords <- seq(1, grid_n)
y_coords <- seq(1, grid_n)
grid <- expand.grid(x = x_coords, y = y_coords)
df_grid <- data.frame(s1 = grid[,1], s2 = grid[,2])

method_dist <- args[4]

if (method_dist == "exponential_times_wave") {
  set.seed(current_seed)
  wave_z1 <- sprnorm(wave_spcov_params(), data = df_grid, xcoord = s1, ycoord = s2)
  set.seed(current_seed)
  df_grid$z1 <- wave_z1 * sprnorm(
    exponential_spcov_params(current_phi),
    data = df_grid, xcoord = s1, ycoord = s2
  )
} else if (method_dist == "gaussian-v2") {
  set.seed(current_seed)
  df_grid$z1 <- sprnorm(
    exponential_spcov_params(current_phi),
    data = df_grid, xcoord = s1, ycoord = s2
  )
} else if (method_dist == "gaussian-nugget-03-v2") {
  set.seed(current_seed)
  df_grid$z1 <- sprnorm(
    spcov_params("exponential", de = 1, ie = 0.3, range = current_phi),
    data = df_grid, xcoord = s1, ycoord = s2
  )
} else {
  base::stop("invalid DISTRIBUTION argument")
}

df_grid$s1 <- df_grid$s1 - 1
df_grid$s2 <- df_grid$s2 - 1

write.csv(df_grid, save_path, row.names = FALSE)
