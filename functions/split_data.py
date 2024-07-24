"""
Contains functionality for splitting data into train and test sets
"""
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

def move_and_split_data():
  """
  Move and split data into train and test sets

  Args:
    None

  Returns:
    test_dir, train_dir: Path to test and train directories
  """
  # split data and move it into respective folders
  def move_files(file_list, destination_folder):
      for file in file_list:
          if file.exists():
            # print(f"File {file} already exists. Overwriting...")
            shutil.copy2(file, destination_folder)
          else:
            shutil.move(file, destination_folder)


  def split_and_move_data(source_folder, train_folder, test_folder, train_size=0.8):
      # Get all file paths
      files = list(Path(source_folder).glob('*.jpg'))

      # Split files into train and test
      train_files, test_files = train_test_split(files, train_size=train_size, random_state=42)

      # Move files to respective folders
      move_files(train_files, train_folder)
      move_files(test_files, test_folder)

  def remove_directory(directory_path):
      if directory_path.exists():
          # Remove directory and all its contents
          shutil.rmtree(directory_path)
          print(f"Removed directory and all its contents: {directory_path}")
      else:
          print(f"Directory {directory_path} does not exist.")

  data_path = Path("data/")
  image_path = data_path / "3D_Defect"

  # Create new directories to make datasets
  train_dir = image_path / "train"
  test_dir = image_path / "test"

  # Create directories
  for path in [train_dir / "defected", train_dir / "not_defected", test_dir / "defected", test_dir / "not_defected"]:
      path.mkdir(parents=True, exist_ok=True)

  print(f"[INFO] Directory structure created.")

  # Paths for defected and not defected datasets
  defected_path = image_path / "defected"
  not_defected_path = image_path / "no_defected"

  # Split and move defected and not defected images
  split_and_move_data(defected_path, train_dir / "defected", test_dir / "defected")
  split_and_move_data(not_defected_path, train_dir / "not_defected", test_dir / "not_defected")

  print(f"[INFO] Data split and moved.")

  remove_directory(image_path / "defected")
  remove_directory(image_path / "no_defected")

  return train_dir, test_dir
