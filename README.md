# Doppelkopf
Here, you can find all the code relating to the Doppelkopf-implementation, the neural networks and their training processes, as well as helper classes for various functions.

**Please note that LSTMs were discontinued, so any class/program/script that deals with LSTMs is not guaranteed to work with the rest of the project, or run at all. I have left them in the Repo for completeness' sake only.**

## Installation
**Before any scripts can be run, the "Doppelkopf" package must be installed. To do so, please nagivate the console/terminal to the following directory:**  
"./src/"  
**The directory should contain a "setup.py", which will handle the installation. Here, enter the command**
```bash
pip install -e .
```
**to install the Doppelkopf package. Make sure to enter the command correctly, including the dot at the end! Please make sure that**
- TensorFlow (preferably version 2.4.1)
- numpy (preferably version 1.19.5)
- colorama (preferably version 0.4.4)

**are installed. Python version 3.8.0 is recommended.**

```bash
py -m pip install colorama
py -m pip install numpy
py -m pip install tensorflow
```

## How to run anything
All executable scripts are found in: "/src/doppelkopf/programs/"  
Some notable examples are:

- "RunServer.py", which starts a Doppelkopf Server instance
- "RunRulebasedPlayer.py", which starts a rule-based Doppelkopf Agent.
- "RunDNNPlayer.py", which starts a DNN, loads weights from a file and plays games using the DNN's estimates.
- "RunDNNTrainerRL.py", which starts the Reinforcement Learning process (Note that this particular program will automatically start other "dummy" players to play with, so there is no need to start those manually.

All these scripts can be run from a CLI, although some may require some arguments to be passed to them. Note that each program blocks the CLI for as long as it is running and might print relevant information to it. I recommend opening multiple terminals or, if not possible, starting screens and running each program in a detached screen:

```bash
screen -S Server -d -m python RunServer.py
```

Screens can be started in detached mode by using the flags `-d -m`. Naming screens is recommended to distinguish between multiple screens running in the background.

### Passing Arguments
All arguments are named, so passing an argument requires the *name* of the argument followed by the *value* of the argument. Arguments can be passed in any arbitrary order, like so:

```bash
python RunServer.py host localhost port 8088
```
or
```bash
python RunServer.py port 8088 host localhost
```

The order does not matter, only that each argument name has a matching value following it. Programs that require certain arguments will tell you so when trying to run them without passing said argument:

> ERROR: ["Missing a required argument: 'example'"]

### Common use cases
1. Common use case number 1: Training a network using Reinformcent Learning  
To train a network, first start a Doppelkopf Server:
```bash
python RunServer.py
```
Without any arguments, the Server will open its socket on "localhost" using port "8088".  
Next, start the training program and specify a few arguments:
```bash
python RunDNNTrainerRL.py numOfGames x saveWeightsPath y
```
where "x" is the desired number of games (this parameter is optional, but it is recommended to set this variable or else the trainer will run indefinitely) and "y" is the relative path to the file, where the neural network's weights are to be stored.

2. Common use case number 2: Running a trained neural network against other players  
To test a DNN's performance, start a Server, start a DNN Agent and let it play against 3 agents of your choosing, for example:
```bash
python RunServer.py
python RunRulebasedPlayer.py # <- Dummy Player 1
python RunRulebasedPlayer.py # <- Dummy Player 2
python RunRulebasedPlayer.py # <- Dummy Player 3
python RunDNNPlayer.py loadWeightsPath x
```
where "x" is the relative (or absolute) path to the file containing the neural network's trained weights. Of course, the DNN Player can also play against other DNN Players, in which case you can easily start one of them using the same command, rather than using the "RunRulebasedPlayer.py" script.  

3. Common use case number 3: Collecting training data for Supervised Learning. To do so, start a Server and play games using the DNNDataRecorderSL:
```bash
python RunServer.py
python RunRulebasedPlayer.py # <- Dummy Player 1
python RunRulebasedPlayer.py # <- Dummy Player 2
python RunRulebasedPlayer.py # <- Dummy Player 3
python RunDNNDataRecorderSL.py trainingPath x evaluationPath y percentage 0.8
```
where "x" and "y" are the relative (or absolute) paths to the files where the training and evaluation data are to be stored, while "percentage" is a float between 0 and 1 indicating how many of the played games are considered "training" data, with the rest automatically becoming "evaluation" data.

**Remember to always start the Server first, otherwise the Agents will terminate upon failing to connect**.

For a comprehensive list of each program's arguments, please have a look at the corresponding .py files. Listing them here would just be tedious. The arguments are listed at the top of each program's .py file, as a list of either "OptionalArgument" instances or "Argument" (required argument) instances. Optionals all have working default values.
