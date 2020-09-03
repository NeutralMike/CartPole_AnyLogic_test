<h1>CartPole_AnyLogic_test</h1>

<h1>Idea</h1>
* Питон скрипт создает и обучает keras модель обучения с подкреплением.
* Пример взят с сайта [keras](https://keras.io/examples/rl/actor_critic_cartpole/).
* Данная модель сохраняется в src/main/resources.
* Оттуда модель загружается при запуске Java программы.
* Java программа взаимодействует с портированной версией gym cartpole (симуляция балансирования шеста) и передает
наиболее выгодные, по оценкам модели, действия для балансирования шеста. 

<h1>Install</h1>
<h4>Java</h4>
mvn install
<h4>python install requirements</h4>
pip install -r python/requirements.txt

<h1>Run</h1>
<h4>Use created model on Java</h4>
mvn clean compile exec:java
<h4>Create and train new model on python</h4>
python python/create_RL_model.py
