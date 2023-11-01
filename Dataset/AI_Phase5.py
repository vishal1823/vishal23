import pandas as pd
import matplotlib.pyplot as plt

def compare_csv_files(True_csv, Fake_csv):
  """Compares two CSV files and returns a DataFrame with the differences."""

  # Read the CSV files into DataFrames.
  true_df = pd.read_csv(True_csv)
  false_df = pd.read_csv(Fake_csv)

  # Get the unique values of the subject column in the two DataFrames.
  true_subjects = true_df['subject'].unique()
  false_subjects = false_df['subject'].unique()

  # Find the subjects that are present in one DataFrame but not the other.
  true_only_subjects = set(true_subjects) - set(false_subjects)
  false_only_subjects = set(false_subjects) - set(true_subjects)

  # Create a DataFrame with the differences.
  diff_df = pd.DataFrame({
      'Subject': list(true_only_subjects) + list(false_only_subjects),
      'Count': [len(true_df[true_df['subject'] == subject]) for subject in true_only_subjects] +
               [len(false_df[false_df['subject'] == subject]) for subject in false_only_subjects]
  })

  return diff_df

def plot_results(diff_df):
  """Plots the results of the CSV file comparison."""

  # Get the subject and count columns from the DataFrame.
  subjects = diff_df['Subject'].tolist()
  counts = diff_df['Count'].tolist()

  # Sort the lists by count.
  sorted_subjects = []
  sorted_counts = []
  for i in range(len(subjects)):
    max_idx = counts.index(max(counts))
    sorted_subjects.append(subjects[max_idx])
    sorted_counts.append(counts[max_idx])
    counts[max_idx] = -1

  # Create a bar chart.
  plt.bar(sorted_subjects, sorted_counts)
  plt.xlabel('Subject')
  plt.ylabel('Count')
  plt.title('Fake News Detection Using NLP: Subjects Only Present in One CSV File')
  plt.show()

# Compare the True and False CSV files.
diff_df = compare_csv_files('True.csv', 'Fake.csv')

# Plot the results.
plot_results(diff_df)
