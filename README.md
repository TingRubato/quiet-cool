
<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Quiet Cool: Intelligent GPU Fan Speed Control. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/tingrubato/quiet-cool">
    <img src="misc/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Quiet Cool: IPMIntellegent GPU Fan Speed Control</h3>

  <p align="center">
    A server application designed to control GPU fan speeds based on temperature readings and machine learning. This is able to dynamically adjust fan speeds to optimize hardware performance and longevity, especially when you are using a GPU that is not supported by the vendor for fan control, like using customer level GPU on a enterprise level server.
    <br />
    <a href="https://github.com/tingrubato/quiet-cool"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/tingrubato/quiet-cool">View Demo</a>
    ·
    <a href="https://github.com/tingrubato/quiet-cool/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/tingrubato/quiet-cool/issues/new?labels=enhancement&template=feature-request

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#key-features">Key Features</a></li>
        <li><a href="#mathematical-model">Mathematical Model</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#api-reference">API Reference</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

Quiet Cool is a server application designed to control GPU fan speeds based on temperature readings. It dynamically adjusts fan speeds to optimize hardware performance and longevity. Below is a figure that shows how it works and you can find the live version [here](https://app.terrastruct.com/diagrams/1382878080). Source code for the figure is available [here](misc/quiet-cool/misc/diagram.d2).

<iframe
    title="quiet cool"
    width="100%"
    height="100%"
    src="https://app.terrastruct.com/diagrams/1382878080">
</iframe>

## Disclaimer
The script in this project will take over the fan control of your homelab server. Brace yourself for a wild ride! Just remember, with great power comes great responsibility... and the potential for some seriously cool airflow. But hey, I must warn you, this operation is like riding a rollercoaster blindfolded. It's a risky adventure that could leave your hardware feeling a bit shaken, not stirred. So, buckle up, hold on tight, and use this script at your own risk. I take no responsibility for any unexpected fan-induced windstorms or hardware mishaps. Happy fan-controlling! 🌪️💨

### Key Features

* **Dynamic Fan Speed Control**: Adjusts GPU fan speeds in real-time based on temperature data.
* **REST API**: Provides an API endpoint for remote temperature data reception and fan speed control.
* **Logging and Monitoring**: Logs detailed information about temperature readings, fan speed adjustments, and system errors.

### Tested on:
* **Dell PowerEdge R720**: A server with a GPU passed through to an ubuntu 22.04 VM.

### Mathematical Model

#### Initial Control Logic

The initial control logic is defined as a mathematical function:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?f(t)=\begin{cases}10&\text{if%20}t<40^\circ%20C\\20&\text{if%20}40^\circ%20C\leq%20t<45^\circ%20C\\30&\text{if%20}45^\circ%20C\leq%20t<50^\circ%20C\\40&\text{if%20}50^\circ%20C\leq%20t<55^\circ%20C\\50&\text{if%20}t\geq%2055^\circ%20C\end{cases}" alt="Initial Control Logic"/>

#### Machine Learning Model

After the initial 100 iterations using the above control logic, a machine learning model is used to predict the optimal fan speed. The model is trained using a custom loss function designed to balance fan noise and GPU temperature, penalizing the model when noise exceeds residential standards or the GPU temperature is too high.

##### Custom Loss Function

The custom loss function $L$ used during training is defined as follows:

$$L = \text{MSE}(f_\text{pred}, f_\text{true}) + \frac{1}{n} \sum_{i=1}^{n} \left[ P_\text{noise}(n_i) + P_\text{temp}(t_i) \right]$$

Where:

* $f_\text{pred}$ is the predicted fan speed.
* $f_\text{true}$ is the actual fan speed.
* $P_\text{noise}(n_i)$ is the penalty for noise, defined as:

$$P_\text{noise}(n_i) = \begin{cases} 200 & \text{if } n_i < 0.5 \\ 1 & \text{otherwise} \end{cases}$$

* $P_\text{temp}(t_i)$ is the penalty for temperature, defined as:

$$P_\text{temp}(t_i) = \begin{cases} 200 & \text{if } t_i < 50^\circ C \\ 1 & \text{otherwise} \end{cases}$$

$$P_\text{noise}(n_i) = \begin{cases} 10 \times (n_i - 55)^2 & \text{if } n_i > 55 \\ 0 & \text{otherwise} \end{cases}$$

* $P_\text{temp}(t_i)$ is the penalty for temperature, defined as:

$$P_\text{temp}(t_i) = \begin{cases} 20 \times (t_i - 85)^2 & \text{if } t_i > 85 \\ 0 & \text{otherwise} \end{cases}$$

* $n_i$ is the predicted noise level in dB.
* $t_i$ is the predicted temperature in °C.
* $n$ is the number of predictions.

#### Getting Started
---------------
### Prerequisites

#### On your Flask Server Machine:
* Python 3.6+
* - flask
* - numpy
* - pandas
* - tensorflow(up to what you have on the machine that runs the flask server)

#### On your GPU-Passed-Through Machine:
* `nvidia-smi` command-line utility on your GPU-Passed-Through machine
* `curl` command-line utility on your GPU-Passed-Through machine to post GPU temperature to the fan control flask server.


### Installation

1. **Clone the Repository**

```bash
    git clone https://github.com/yourusername/quiet-cool.git
    cd quiet-cool
```

2. **Set Up Python Environment**

You really should use a virtual environment to avoid conflicts with other Python projects. Here's how to set up a virtual environment using `venv`:

```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
```

3. **Run the Server**

```bash
    python fan_control_server.py
```

<!-- USAGE EXAMPLES -->
## Usage

Use the script `monitor_gpu_temp.sh` to send GPU temperatures to the server:

```bash
#!/bin/bash
# monitor_gpu_temp.sh: Script to monitor GPU temperature and send it to a remote machine

REMOTE_ADDRESS="http://your_flask_server:23333"  # Replace with your remote machine's address

get_gpu_temp() {
        nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits
}

while true; do
        TEMP=$(get_gpu_temp)
        echo "GPU Temperature: $TEMP°C"
        curl -X POST -d "temperature=$TEMP" $REMOTE_ADDRESS/gpu-temperature
        sleep 10
done
```

<!-- API REFERENCE -->
## API Reference

**POST `/gpu-temperature`**

- **Request**: `temperature` - current GPU temperature.
- **Response**: `200 OK` - fan speed adjusted.

<!-- FUTURE WORK -->

## Future Work
- [ ] Remodel the noise level calculation to include more factors by diving deeper into the specifics of the fan module.
- [⏳] Research on the possiblity of using different fan speed control Mode. E.g. quiet mode, performance mode, responsive mode etc.
- [ ] Start the frontend user interface development to allow user to modify settings and add some observability to the user interface. (I just kick started node.js courses and let's see how long will it take.)
- [ ] Support more Machines.

<!-- KNOWN ISSUES -->
## Known Issues

### Noise Level Calculation

The current noise level calculation is based on a simplified model and may not accurately reflect the actual noise produced by the fan module, resulting in suboptimal fan speed adjustments. Ideally, the noise level should be calculated using the actual fan speed retrieved from iDRAC and the specifications of the fans. However, as I am not an acoustic engineer, I will temporarily assume that this calculation is sufficient until I find a better solution. Any suggestions are welcome.

### Mathematical Model Limitations

Currently, the penalization of noise and temperature is based on a simplified quadratic function. However, this approach may not accurately capture the real-world impact of noise and temperature on user experience and hardware longevity. At the moment, my focus is on timely adjustments, which is why the penalization is set to be somewhat aggressive. In future versions, I plan to refine these models based on more detailed research and valuable user feedback.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes or improvements.

<!-- LICENSE -->
## License

This project is available under the MIT License. See [LICENSE.md](LICENSE.md) for more details.

<!-- CONTACT -->
## Contact

For any questions or collaborations, feel free to reach out via email:

- **Ting Xia**
  - 📧 Email: [tingrubato@outlook.com](mailto:tingrubato@outlook.com)

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [serverManager](https://github.com/Danielv123/serverManager)
* [d2lang](https://d2lang.com)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
