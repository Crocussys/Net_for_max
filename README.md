<h1>Описание</h1>
<p>Настраиваемая нейронная сеть.<br>
Выходные значения принадлежат интервалу (0, 1).<br>
Метод обучения - Обратного распространения ошибки (backpropagation).<br>
Ошибка - среднеквадратичная (MSE).<br>
Функция активации - сигмоида (sigmoid).<br>
Вход-выход через csv формат.</p>
<h1>Установка</h1>
<ol>
  <li>Открыть терминал Widows/PowerShell/Unix</li>
  <li>Перейти в папку, куда будет скопирован репозиторий<br>В Unix, например, это $ cd [path]</li>
  <li>$ git clone https://github.com/Crocussys/Net_for_max.git</li>
</ol>
<h1>Запуск</h1>
<ol>
  <li>Перейти в папку репозитория (если вы не там)<br>$ cd [path]</li>
  <li>$ python3 main.py<br>Аргументы командной строки не требуются</li>
</ol>
<h1>Настройка config.json</h1>
<table>
  <thead>
    <tr>
      <th>Параметр</th>
      <th>Тип</th>
      <th>Описание</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>create</td>
      <th>bool</th>
      <td>true - При запуске будет создан новый файл нейронной сети.<br>false - При запуске попытается открыть существующий файл, и работать с ним</td>
    </tr>
    <tr>
      <td>learn</td>
      <th>bool</th>
      <td>true - После открытия файла начнётся обучение нейронной сети (по файлу learning.csv).<br>false - Этого не происходит</td>
    </tr>
    <tr>
      <td>inference</td>
      <th>bool</th>
      <td>true - После открытия/обучения прогонит по нейронной сети данные из файла input.csv.<br>false - Этого не происходит</td>
    </tr>
    <tr>
      <td>plot</td>
      <th>bool</th>
      <td>true - Вкл. отображение графиков.<br>false - Выкл. отображение графиков</td>
    </tr>
    <tr>
      <td>path</td>
      <th>string</th>
      <td>Путь до папки с файлами network.json, learning.csv, input.csv, output.csv</td>
    </tr>
    <tr>
      <td>config_network</td>
      <th>int array</th>
      <td>Конфигурация нейронной сети. Т.е. Количество входов -> количество нейронов в слое -> ... -> количество выходов</td>
    </tr>
    <tr>
      <td>learn_rate</td>
      <th>int</th>
      <td>Скорость обучения. Коэфициент в градиентном спуске</td>
    </tr>
    <tr>
      <td>epochs</td>
      <th>int</th>
      <td>Количество эпох. Т.е. сколько раз прогнать все значения при обучении</td>
    </tr>
    <tr>
      <td>print_every</td>
      <th>int</th>
      <td>Показывать результат обучения каждые print_every эпох.</td>
    </tr>
</table>
<h1>Папка с файлами </h1>
<ul>
    <li>network.json - Файл нейронной сети. <font color="red">НЕ ИЗМЕНЯЙТЕ ЭТОТ ФАЙЛ</font> - может отсутствовать</li>
    <li>learning.csv - Файл данных для обучения (раздерелитель ";") - может отсутствовать, если learn = false</li>
    <li>input.csv - Файл данных для пропуска через нейронную сеть без обучения (раздерелитель ";") - может отсутствовать, если inference = false</li>
    <li>output.csv - Выход из нейронной сети для данных из input.csv (раздерелитель для output ",") - может отсутствовать</li>
</ul>
<p><font color="red">ПРОВЕРЬТЕ</font> чтобы config_network в config.json соответствовал learning.csv и input.csv. Т.е. learning.csv и input.csv должны содержать (количество входов + количество выходов) столбцов. Количество строк не ограничено.<br>
Если вам не известны выходы в input.csv занулите их и укажите параметр plot в config.json равным false</p>
<h1>Для опытных пользователей</h1>
<p>В файле NeuralNet.py можно изменить:</p>
<ul>
    <li>функцию активации (activation, line 6)</li>
    <li>функцию ошибки (loss, line 10)</li>
    <li>значение step - отступ для численного дифференцирования (self.step, line 16)</li>
</ul>
<p>Можно изменить отображение графиков (NeuralNet.py, line 91) (main.py, line 42)</p>
<h1>Заключение</h1>
<p>Bug report, вопросы, предложения, отзывы через <a href="https://t.me/crocus_sys">телеграм</a></p>