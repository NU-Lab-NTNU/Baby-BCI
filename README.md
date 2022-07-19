# NU-BCI-v4
WARNING:
  - The project is currently in an unfinished, minimally tested, state and in rapid development

The goal of this project is twofold:
  - Create an interface with EGI AmpServer with as little unnecessary complexity as possible.
  - Create a BCI-system which is easily modifiable and accessible to non-programmers (ex. psychology scientists, ML-people).
 
Implementation: 
  - The project relies on standard python libraries for ease of use.
  - The project is designed with our lab's experimental setup in mind: data stream from NA300 amplifier, stimulus presentation done by E-Prime. Both on external computers.
  - Each task, managing data stream, communicating with E-Prime and performing signal processing, is done by different submodules, each with their dedicated thread for their main loop. The submodules are coordinated by the operator loop.
  
 TODO:
  - Add UI
  - Error handling
  - Create general parent class for submodules
  
Notes:
  - The AmpServerClient module is heavily inspired by [labstreaminglayer/App-EGIAmpServer](https://github.com/labstreaminglayer/App-EGIAmpServer)
