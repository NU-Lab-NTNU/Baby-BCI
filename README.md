[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# NU-BCI

WARNING:
  - The project is currently in an unfinished, minimally tested, state and in rapid development. Releases will be published whenever the author feels somewhat confident in the performence and fidelity of the system.
  - In order to read real-time data stream from AmpServer (through AmpServerClient.py), [AmpServer Pro SDK](https://www.egi.com/images/stories/placards/ASProSDK_v21_ugid_8409500-57_20181029.pdf) is required.
 
What this project is:
  - The primary goal of this project is to create a bci-system fit for our lab's needs:
    * Real time EEG data from EGI NetAmps amplifiers (specifically NA300).
    * Two-way communication with E-Prime (stimulus presentation software).
    * Allow for inter-trial classification.
    * Enable my successors to focus on neuropsychology research and ML-algorithms rather than multithreading and TCP sockets.
  - Secondary goals include:
    * Provide inspiration, possibly even a working framework, to other labs seeking to create a bci-system using the same or similar technologies.
    * Expand flexibility by providing support for other EGI NetAmps series.
    * Hide as much low-level stuff as possible in order to improve accessibility.
  - Technologies used:
    * Standard python libraries are used almost exclusively, the only exception being numpy.
 
What this project is not:
  - A general BCI framework.
    * This project was started since other BCI frameworks for some reason did not offer the functionality our lab needed. If a general framework is what you need, there are other, probably more suitable projects, such as [BCI2000](https://www.bci2000.org/mediawiki/index.php/Main_Page) or [Timeflux](https://timeflux.io/)
 
 Usage:
  - Installation:
    ```
    git clone https://github.com/vegardkb/NU-BCI.git
    cd NU-BCI/
    pip install -r requirements.txt
    ```
  - How to run first time:
    ```
    python config/generate_config.py
    python CommandLineInterface.py
    ```
    If you'd also like to see the debug log open a terminal or command prompt, cd into NU-BCI and try:
    ```
    tail -f log/debug.log
    ```
    or if tail doesn't work(windows):
    ```
    powershell
    Get-Content log/debug.log -Wait
    ```
 
Timeline:
  - Future plans:
    * v1.0.0 Command line interface

This project is worked on by NTNU's NuLab:
  - [In-house website](https://nulab-ntnu.github.io/)
  - [NTNU's website](https://www.ntnu.edu/psychology/nulab)
  
Notes:
  - The AmpServerClient module is heavily inspired by [labstreaminglayer/App-EGIAmpServer](https://github.com/labstreaminglayer/App-EGIAmpServer)
  - This project is the successor to https://github.com/dermanu/NU-BCI
