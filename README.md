# BetterSweeper

![g](icon.ico)

An improved iteration of my sweeperbot written in Python using various techniques/algorithms. Download the latest release [here](https://github.com/kymotsujason/bettersweeper/releases/latest) to try it out yourself. It's compatible with the Microsoft Minesweeper from the Microsoft Store and tested with default settings (default theme, all settings on). 2560x1440 is ideal res, but 1920x1080 should be fine too. It technically looks slower than my previous ones, but that's because of animiations. It could be faster if adjusted for the old one, but I wanted to target a more modern version.

[![YouTube](http://i.ytimg.com/vi/H8aDuFTcfJs/hqdefault.jpg)](https://www.youtube.com/watch?v=H8aDuFTcfJs)

Some notable functions are:

- solve_board_state:
  - Builds constraints (rule indicating mines adjacent to a tile) with bitmasked neighbors (optimised representation of adjacents)
  - Inference based on the constraint (if all adjacents are safe or mines) and then propagated through neighbouring constraints
  - Optimisations include: Bitmask (super fast comparisons), Caching (saving completed processing of neighbors), Memoisation (saving the discovering of neighbors), Sorting constraints (less comparisons)
  - If there are no safe moves, call choose_least_risky_hidden_cell which calls estimate_cell_probability (smart guessing, but there's a limit to how smart we can be)

- estimate_cell_probability:
  - risk = (number − flagged_neighbors) / (total hidden neighbors)
  - safe = 1 − risk
  - w = 1 / (total_hidden_neighbors * distance) where distance is the Euclidean distance between (i,j) and the neighbor's cell.
  - combined_safe = exp( (Σ w * ln(safe)) / (Σ w) )
  - estimated_risk = 1 − combined_safe
  - Basically, calculate risk from neighboring numbers (mine count) and weigh neighbors with less hidden cells more

- get_board_state:
  - Uses mss to capture screenshots
  - Crops the titlebar/navbar/sidebar
  - Calls detect_board_cells
  - Calls classify_cell_by_color

- detect_board_cells:
  - Gray scale > blur > contour detection > filtering > [non maximum suppression](https://www.geeksforgeeks.org/what-is-non-maximum-suppression/) > group vertically

- classify_cell_by_color:
  - Samples the sweet spot or patch of pixels and compares the color to the color map

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to utilize this project and how to install them

```
Python v3.13.2 - https://www.python.org/
```

### Installing

A step by step series of examples that tell you how to get a development env running

Clone the git

```
git clone https://github.com/kymotsujason/bettersweeper.git
```

Activate virtual environment (vscode might do it automatically, just restart the terminal)

```
./venv/Scripts/Activate.ps1
```

Install the required packages

```
pip install -r requirements.txt
```

Run the script

```
python .\main,py
```

## Deployment

Run this code to create an exe

```
pyinstaller.exe --clean --onefile --name=BetterSweeper --version-file "file_version_info.txt" --icon=icon.ico --add-data "icon.ico;." main.py
```

If you don't have a .spec file in your directory, this will create it and you will need to add this to the .spec file:

```
a.binaries replace with a.binaries + [('icon.ico', 'icon.ico', 'DATA')],
```

Run the code again and it should work this time

```
pyinstaller.exe --clean --onefile --name=BetterSweeper --icon=image.ico main.py
```

## Built With

- [Python](https://www.python.org/) - The language used

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/kymotsujason/bettersweeper/tags).

## Authors

- **Jason Yue** - *Initial work* - [kymotsujason](https://github.com/kymotsujason)

See also the list of [contributors](https://github.com/kymotsujason/bettersweeper/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
