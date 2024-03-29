## Guia de instalación

Nota: Esta guia ha sido realizada para sistemas operativos GNU/LINUX. Hasta el momento los algoritmos no han sido probados en sistemas operativos OSX y Windows, aunque el proceso debe ser muy similar.

### Requerimientos:

    • Python 3.5 (o posterior)

    • Supercollider 3.10 (o posterior)

    • Tensorflow_model_server:

      apt-get update && apt-get install tensorflow-model-server

    • Pip3: PythonOSC, JSON, Essentia, SoundCard_

      pip3 install essentia SoundCard jsonlib python-osc tensorflow pydub toolz

### 1. Clonar desde la terminal o descargar el repositorio:

    git clone https://github.com/Atsintli/Interdictum-Prothesium.git
   
    
### 2. Abrir la terminal en la carpeta del repositorio y ejecutar:

       source init.bash
       
Este código corre paralelamente los 6 algoritmos de escucha automática para comunicarse con Supercollider via OSC.

#### Algoritmos de escucha automática:

    • 05_extract_features_in_realtime+OSC_sender.py

### 3. Abrir el siguiente código con Supercollider y ejecutar (ctrl+enter) todas las lineas de código:
Este código carga mediante el archivo get_audios_&_syths_IP.scd varios sinstetizadores y directorios a los archivos de audio previamente reubicados (es necesario cambiar todos los directorios de get_audios_&_syths_IP.scd a los tuyos propios).

       osc_sonification_data_reciver.scd

### 4. A improvisar con SEALI!!!
       
## Motivaciones

“No pongas jamás tu creación al servicio de otra cosa que no sea la libertad.” Esta es la frase con la que el cineasta checo Jan Savankmajer (Praga, 1934) finaliza su decálogo sobre su particular manera de hacer cine. ¿Cómo intento yo buscar esa libertad? En mi quehacer como creador musical y sonoro he encontrado distintas aproximaciones que me han permitido ampliar mi panorama de acción más allá de las técnicas y prácticas clásicas que involucran los procesos de creación musical, una de estas fue mi acercamiento a las ciencias de la complejidad. A partir de este paradigma derivé una propuesta de creación sonora basada en el estudio de sistemas interrelacionados compuestos por agentes con funciones artificiales que en interacción con ellos mismos y su entorno son capaces de producir resultados emergentes, no previstos a través de mecanismos homeostáticos, es decir mecanismos de adaptación interna a un contexto variable, inestable, cambiante. 

Mi búsqueda hacia aproximaciones no tan convencionales para la creación musical, por una parte, intenta hacer cuestionamientos como ¿de que forma podrían ser modificadas las técnicas y formatos de presentación anclados a la tradición de concierto para la creación musical? ¿cuales serían los nuevos paradigmas que nos permitirían cambiar un orden establecido en la escucha y creación de los productos musicales y sonoros ligados a las prácticas académicas? ¿qué implica la utilización de estos nuevos paradigmas? 

La propuesta que hago, puede ser entendida como un sistema de interacciones que incluye la realización de una máquina interactiva basada en el conocimiento de un corpus musical. Al emplear  herramientas como el  aprendizaje  automático y la escucha de máquinas, la máquina podría recibir entradas sonoras producidas por agentes humanos (no necesariamente músicos) y entorno. Partiendo de estas entradas la máquina podría tomar decisiones a nivel sonoro que se verían reflejas en un bucle interactivo sonoro entre máquina, humano(s) y entorno. En términos sistémicos la propuesta parte de cinco aspectos fundamentales:

Variables de entrada. Una escucha humana que modela y determina el conocimiento musical de la máquina, esto involucra la creación de una base de datos anotada. 

Variables de estado.  Técnicas de aprendizaje de máquinas; extracción de atributos de la base de datos, su clasificación mediante algoritmos como el aprendizaje profundo o las redes neuronales y la creación de un modelo capaz de predecir nuevas instancias nunca antes vistas (esta parte podría ser entendida como un subsistema en si misma, ya que un modelo de aprendizaje profundo recibe entradas que a su vez son codificadas numéricamente por varias capas neuronales escondidas que organizan y clasifican la información dando como salida otra capa que puede ser expresada en forma simbólica, por ejemplo a través de palabras). 

Mecanismos perceptores. Mecanismos de extracción de atributos, reconocimiento y clasificación al momento de nuevas instancias sonoras.

Mecanismos actuadores. Las técnicas algorítmicas necesarias para la generación sonora de la máquina partiendo de lo que reconoce, esto esta ligado con su parte reactiva.

Entorno: Todos los estímulos sonoros externos que la máquina es capaz de percibir, estos incluyen las acciones sonoras de los agentes humanos involucrados.   
 
Esta máquina que escucha y que crea sonidos idealmente sería una ampliación de mí mismo, de mi propia forma de escuchar, hacer y concebir la música. En los últimos años, mi práctica musical se ha centrado en procesos de creación colectiva a través de la libre improvisación, he explorado con sistemas interactivos la creación de música electroacústica, y la creación de piezas con prácticas ligadas a la escucha y la grabación de campo. Además, la música instrumental que me sigue  estimulando  fue creada por compositores como: Maiguashca, Billone, Ablinger, Xenakis, Ferneyhough, Nancarrow, Cowel, Nono, Kampela, Lachenmann, Dumitrescu, entre otros.  Estas aproximaciones a lo sonoro (la improvisación libre, la composición y la grabación de campo) siguen presentes dentro de mi quehacer como creador musical, es por ello que la base de datos que propongo integrar al conocimiento de la máquina incluye estos referentes. 

Asimismo lo que intento hacer con este sistema parte de integrar ciertas estéticas ligadas a cada una de estas prácticas para tratar de encontrar una o varias formas de creación sonora que se encuentren en el intersticio de las mismas. En este sentido el sistema es pensado como un espacio que promueve la interacción colectiva entre distintos agentes humanos y máquinas enfocado en la creación de música nueva donde el objeto sonoro pueda ser entendido como una unidad con cualidades polisémicas según su contexto.

## SEALI 

La parte técnica de este proyecto realizada hasta ahora se ha concentrado en dos grandes vertientes: la sistematización de mi escucha que involucra un proceso de clasificación de música y la obtención de conocimiento para el proceso de aprendizaje automático que incluye, la extracción óptima de atributos  de la base de datos y los procesos automáticos de clasificación. 

Para la clasificación de los fenómenos sonoros y musicales con los que deseo trabajar he indagado trabajos vinculados con autómatas celulares en particular de Christopher Langton y Stephen Wolfram (científicos de la computación y físico respectivamente) quienes introducen algunas reglas generales de comportamiento que pueden exhibir los autómatas celulares.

En el libro “A New Kind of Science” Stephen Wolfram introduce cuatro reglas generales que pueden presentar los autómatas celulares, estos pueden contener reglas de operación relativamente simples que en interacción con todos sus componentes internos su comportamiento general puede llegar a presentar una complejidad creciente que tiende a la generación de patrones regulares. Wolfram señala que estas cuatro reglas generales tienen el potencial de ser extrapoladas a fenómenos naturales en el entendido  que la naturaleza presenta elementos simples que interconectados dan lugar a una serie de fenómenos emergentes complejos. Su tesis apunta a que los sistemas complejos en la naturaleza operan de forma similar a los autómatas celulares. Estas cuatro reglas generales de comportamiento de los autómatas celulares son presentadas de acuerdo con su creciente complejidad.

Wolfram señala que la clase 1 tiene un comportamiento muy simple y por lo general todas las condiciones iniciales llevan al sistema a un estado de uniformidad. Extrapolando esta idea a un lenguaje musical o sonoro encuentro esa uniformidad en los pedales, en sonidos estables que no cambian en el tiempo de acuerdo con una percepción más holística o general del sonido. Un ejemplo de esta clase podría ser el ruido blanco que aunque internamente esta cambiando constantemente, para mi percepción presenta un comportamiento continuo y estable a pesar de sus cambios aleatorios intrínsecos en su producción. Otro ejemplo podría ser una onda sinusoidal la cual pese a que en su estructura interna es periódica tiene un resultado continuo para la percepción. 

La clase 2 “tiene muchos posibles estados finales diferentes, pero todos consisten solo en ciertos conjuntos de estructuras simples que permanecen iguales por siempre o se repiten cada cierto número de pasos”. Trasladado a la música, serían sonidos que se repiten de manera intermitente, presentan una recurrencia constante, hay patrones claros, repetitivos y cíclicos. Independientemente del timbre lo que me interesa definir como periódico esta más relacionado con un estado de reiteración, de igual manera la amplitud podría no ser considerada ni las alturas, sino solamente la esencia rítmica principal que produce la recursividad.   

En la clase 3 “el comportamiento es más complicado, y parece aleatorio en muchos aspectos , sin embargo triángulos y otras estructuras a pequeña escala se ven en algún nivel.” A nivel musical encontraríamos diferentes estructuras lógicas sucediendo al mismo tiempo, además la clase caótica presenta cierta organización que no necesariamente es aleatoria sino difícil de asir, así como en la figura tres podemos ver ciertas regularidades en las formas de los triángulos, en la música éstas regularidades están presentes en el tempo, la intensidad, en la densidad sonora y en muchas ocasiones podemos encontrar comportamientos politímbricos  y polimétricos. Para concretar estas ideas, el caos se encuentra en sistemas dinámicos aperiódicos altamente sensibles a las condiciones iniciales. Los fenómenos caóticos para nuestra percepción podrían ser aleatorios pero no constituyen hechos propiamente aleatorios ya que son el resultado de dinámicas deterministas.

Finalmente la clase 4, “presenta una mezcla entre orden y aleatoriedad: Se producen estructuras localizadas que, por sí mismas, son bastante simples, pero estas estructuras se mueven e interactúan entre sí de formas muy complicadas.”1 En el contexto musical el comportamiento de la clase compleja estaría en la frontera entre varias clases ya sea periódica y fija o caótica y fija o periódica y compleja, es decir varios componentes diferentes interactuando entre si de manera simultanea. Este tipo de comportamiento puede ser visto en el cuarto recuadro de las imágenes antes mostradas donde podemos apreciar lineas rectas con cierta regularidad, patrones fijos como los representados por el color negro y además estructuras bastante sofisticadas que pareciera generan un orden de tipo fractal. El contrapunto entre dos o más tipos de materiales puede dar lugar a una amalgama de clases, donde podríamos tener una clase periódica que, por su naturaleza tendente a fija se vuelve compleja. La clase compleja tiene una tendencia contrapuntista entre las clases fija, periódica y caótica.

La potencialidad que encuentro al emplear estas categorías, que en un principio podrían parecer muy generales, es que me permiten aproximarme de forma sencilla a un problema que podría ser inmensamente complejo como es la clasificación manual de una base de datos. Además esta categorización me es útil, ya que los resultados obtenidos (mostrados más adelante) me han sido útiles para los fines perseguidos. No estoy diciendo que esta clasificación sea la única forma de clasificar  la música a la que me aproximo, sino que puede haber muchas otras formas para clasificarla. De las diferentes formas de clasificación es posible derivar resultados completamente diferentes que incluso pueden ser utilizados para producir resultados estéticos diferentes. 
