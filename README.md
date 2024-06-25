**Quiet Cool: Intelligent GPU Fan Speed Control**
=====================================================

Quiet Cool is a server application designed to control GPU fan speeds based on temperature readings. It dynamically adjusts fan speeds to optimize hardware performance and longevity.

**Key Features**
---------------

* **Dynamic Fan Speed Control**: Adjusts GPU fan speeds in real-time based on temperature data.
* **REST API**: Provides an API endpoint for remote temperature data reception and fan speed control.
* **Logging and Monitoring**: Logs detailed information about temperature readings, fan speed adjustments, and system errors.

**Mathematical Model**
--------------------

### Initial Control Logic

The initial control logic is defined as a mathematical function:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?f(t)=\begin{cases}10&\text{if%20}t<40^\circ%20C\\20&\text{if%20}40^\circ%20C\leq%20t<45^\circ%20C\\30&\text{if%20}45^\circ%20C\leq%20t<50^\circ%20C\\40&\text{if%20}50^\circ%20C\leq%20t<55^\circ%20C\\50&\text{if%20}55^\circ%20C\leq%20t<60^\circ%20C\\60&\text{if%20}60^\circ%20C\leq%20t<65^\circ%20C\\70&\text{if%20}65^\circ%20C\leq%20t<70^\circ%20C\\80&\text{if%20}70^\circ%20C\leq%20t<75^\circ%20C\\90&\text{if%20}75^\circ%20C\leq%20t<80^\circ%20C\\100&\text{if%20}t\geq%2080^\circ%20C\end{cases}" alt="Initial Control Logic"/>
</p>

### Machine Learning Model

After the initial 100 iterations using the above control logic, a machine learning model is used to predict the optimal fan speed. The model is trained using a custom loss function designed to balance fan noise and GPU temperature, penalizing the model when noise exceeds residential standards or the GPU temperature is too high.

#### Custom Loss Function

The custom loss function `L` used during training is defined as follows:


<p align="center">
<img src="https://latex.codecogs.com/svg.latex?L%20=%20\text{MSE}(f_{\text{pred}},%20f_{\text{true}})%20+%20\frac{1}{n}%20\sum_{i=1}^{n}%20\left[%20P_{\text{noise}}(n_i)%20+%20P_{\text{temp}}(t_i)%20\right]" alt="Custom Loss Function"/>
<br>
Where:<br>
- \(f_{\text{pred}}\) is the predicted fan speed.<br>
- \( f_{\text{true}} \) is the actual fan speed.<br>
- \( P_{\text{noise}}(n_i) \) is the penalty for noise, defined as:<br>
<img src="https://latex.codecogs.com/svg.latex?P_{\text{noise}}(n_i)%20=%20\begin{cases}%200&%20\text{if%20}%20n_i%20<%200.5\\%201&%20\text{otherwise}%20\end{cases}" alt="Penalty for Noise"/>
<br>
- \( P_{\text{temp}}(t_i) \) is the penalty for temperature, defined as:<br>
<img src="https://latex.codecogs.com/svg.latex?P_{\text{temp}}(t_i)%20=%20\begin{cases}%200&%20\text{if%20}%20t_i%20<%2050^\circ%20C\\%201&%20\text{otherwise}%20\end{cases}" alt="Penalty for Temperature"/>
</p>

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?P_{\text{noise}}(n_i)%20=%20\begin{cases}%2010%20\times%20(n_i%20-%2055)^2%20&%20\text{if%20}%20n_i%20>%2055%20\\%200%20&%20\text{otherwise}%20\end{cases}" alt="Noise Penalty"/>
</p>

- \( P_{\text{temp}}(t_i) \) is the penalty for temperature, defined as:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?P_{\text{temp}}(t_i)%20=%20\begin{cases}%2020%20\times%20(t_i%20-%2085)^2%20&%20\text{if%20}%20t_i%20>%2085%20\\%200%20&%20\text{otherwise}%20\end{cases}" alt="Temperature Penalty"/>
</p>

- \( n_i \) is the predicted noise level in dB.
- \( t_i \) is the predicted temperature in °C.
- \( n \) is the number of predictions.

The loss function ensures that the model not only learns to predict accurate fan speeds but also maintains acceptable noise levels and safe GPU temperatures.

**Getting Started**
---------------

### Prerequisites
* Python 3.6+
* `nvidia-smi` for NVIDIA GPU temperature monitoring

### Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/quiet-cool.git
    cd quiet-cool
    ```

2. **Set Up Python Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. **Run the Server**

    ```bash
    python fan_control_server.py
    ```

**Usage**
-----

Use the script `monitor_gpu_temp.sh` to send GPU temperatures to the server:

```bash
#!/bin/bash
# monitor_gpu_temp.sh: Script to monitor GPU temperature and send it to a remote machine

REMOTE_ADDRESS="http://192.168.100.15:23333"  # Replace with your remote machine's address

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

**API Reference**
---------------

**POST `/gpu-temperature`**

* **Request**: `temperature` - current GPU temperature.
* **Response**: `200 OK` - fan speed adjusted.

**Contributing**
------------

Contributions are welcome. Please open an issue to discuss proposed changes or improvements.

**License**
-------

This project is available under the MIT License. See [LICENSE.md](LICENSE.md) for more details.
