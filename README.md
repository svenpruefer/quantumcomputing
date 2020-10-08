# Quantum Space Operations Center

Copyright (c) 2020 by DLR.

[![Build Status](https://travis-ci.org/svenpruefer/quantumcomputing.svg?branch=master)](https://travis-ci.org/svenpruefer/quantumcomputing)

This is a repository containing project files of the QSOC (_Quantum Space Operations Center_) for the Quantumcomputing Workshops organized
by Ludwig-Maximilians-Universität, Universität der Bundeswehr and Deutsches Zentrum für Luft- und Raumfahrt e.V.
 
 So far, this repository contains code for the following workshops:
  - March 2020: Grover algorithm and graph coloring problems
  - October 2020: error correction

**If you are participating as a student, do NOT look further as this repository may contain solutions to the problem sheets.**

_Seriously, don't cheat! :-)_

## Setup

Prerequisites:
* At least Python 3.8. However, it might also work under slightly older versions.

Initial Setup:
1. Clone this repository from e.g. [https://gitlab.dlr.de/qc-gsoc/qsoc]() or [https://gitlab.lrz.de/qc-hackathon/team-dlr]().
2. Create a virtual environment (e.g. [venv](https://docs.python.org/3/library/venv.html)) either by following the instructions on their website or by using an IDE.
3. Run `pip install -r requirements.txt` within your virtual environment to install the dependencies.

## Testing

Tests are run by executing e.g.

```shell
$ pytest
$ pytest qsoc/tests/unit_tests/
```
