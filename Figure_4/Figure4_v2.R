# Python part
# adata = ep.io.read_h5ad("/Users/tim.treis/Documents/GitHub/ehrapy-reproducibility/adata_pneumonia_unspecified_rest_annotated.h5ad")
# data = adata.to_df()
# data["group"] = adata.obs["Pneumonia unspecified - annotated"]
# data.to_csv("/Users/tim.treis/Documents/GitHub/ehrapy-reproducibility/adata_pneumonia_unspecified_rest_annotated.csv")


library("tidyverse")
library("ggplot2")


# read data
data <- read.csv(
  "/Users/tim.treis/Documents/GitHub/ehrapy-reproducibility/adata_pneumonia_unspecified_rest_annotated.csv",
)

# replace categories by correct strings
data = data %>% 
  dplyr::mutate(group = dplyr::case_when(
    group == "viral pneumonia, mild course" ~ "viral pneumonia (n=97)",
    group == "sepsis" ~ "sepsis-like pneumonia (n=28)",
    group == "mild bacterial pneumonia" ~ "mild bacterial pneumonia (n=78)",
    group == "severe bacterial pneumonia with fungal co-infection" ~ "severe pneumonia with co-infection (n=74)",
    TRUE ~ group  # Keeps other values as-is
))

# subset to columns I want to use and reshape
cols_to_plot = c(
  "Asparate.Aminotransferase..AST._max",
  "Alanine.Aminotransferase..ALT._max",
  "Gamma.Glutamyltransferase_max",
  "Bilirubin..Indirect_max",
  "Bilirubin..Direct_max",
  "Bilirubin..Total_max",
  "Potassium_max",
  "Creatinine_max",
  "Albumin_min",
  "LOS"
)
data_long <- data %>% 
  tidyr::pivot_longer(cols = all_of(cols_to_plot), names_to = "variable", values_to = "value")


# define custom colors and panel labels
custom_colors <- c(
  "viral pneumonia (n=97)" = "#d72428", 
  "severe pneumonia with co-infection (n=74)" = "#29a137", 
  "sepsis-like pneumonia (n=28)" = "#f07e1b", 
  "mild bacterial pneumonia (n=78)" = "#1c78b5"
)

custom_labels <- as_labeller(c(
  `Asparate.Aminotransferase..AST._max` = "AST max",
  `Alanine.Aminotransferase..ALT._max` = "ALT max",
  `Gamma.Glutamyltransferase_max` = "GGT max",
  `Bilirubin..Indirect_max` = "Bilirubin indirect max",
  `Bilirubin..Direct_max` = "Bilirubin direct max",
  `Bilirubin..Total_max` = "Bilirubin total max",
  `Potassium_max` = "Potassium max",
  `Creatinine_max` = "Creatinine max",
  `Albumin_min` = "Albumin max",
  `LOS` = "Length of stay"
))


# dummy data for spots as legend
legend_data <- data.frame(
  group = names(custom_colors),
  color = custom_colors
)


# plot
p <- ggplot(data_long, aes(x = value, color = group)) + 
  geom_density(aes(group = group), size = 1, show.legend = FALSE) +
  geom_point(data = legend_data, aes(y = Inf, x = Inf, color = group, fill = group), shape = 21, size = 0) + # size = 0 to hide spot
  scale_color_manual(values = custom_colors) +
  scale_fill_manual(values = custom_colors) +
  guides(color = guide_legend(override.aes = list(shape = 21, size = 3))) +  # size = size in legend
  facet_wrap(~variable, scales = "free",labeller = custom_labels) +
  labs(
    x = "log values",
    y = "Density"
  ) +
  theme_classic()

# Add a line segment to mimic the right axis
p <- p + geom_segment(aes(x = Inf, y = -Inf, xend = Inf, yend = Inf), lineend = "butt", color = "black")

p

