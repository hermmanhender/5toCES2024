# OPTIMIZACIÓN DEL CONFORT TÉRMICO Y CONSUMO ENERGÉTICO EN VIVIENDAS BIOCLIMÁTICAS MEDIANTE APRENDIZAJE POR REFUERZOS PROFUNDO

## RESUMEN

El sector residencial en Argentina tiene un elevado consumo de recursos energéticos y está estrechamente relacionado con las emisiones de gases de efecto invernadero. Esto indica la necesidad imperiosa de optimizar los sistemas energéticos en las viviendas. Aunque las estrategias bioclimáticas son una opción viable, los ahorros esperados no siempre se logran debido al comportamiento del usuario, que puede llevar a una operación no óptima de los mecanismos de climatización del hogar. Una solución a esta problemática es la automatización de ciertos componentes.

Este trabajo presenta un modelo de control basado en aprendizaje por refuerzos profundo, entrenado con un algoritmo de gradiente de Optimización de Políticas Próximas (PPO) y el uso de memoria a largo plazo (LSTM). Este modelo se aplica a un entorno simulado en EnergyPlus de una vivienda bioclimática de interés social en la provincia de Mendoza. La metodología propuesta introduce nuevos enfoques para obtener políticas óptimas que se adaptan mejor a los escenarios planteados. Estos escenarios incluyen uno en el que se controla el encendido y apagado del sistema de acondicionamiento de aire frío/calor, otro en el que se regulan los niveles de temperatura requeridos en el espacio interior para mejorar el confort de los habitantes, y finalmente uno que controla, además de las temperaturas requeridas, el flujo másico de aire refrigerado.

Los escenarios presentan diferencias en cuanto a consumo energético y confort en comparación con un control de referencia convencional basado en reglas. También se observan diferencias en la operación obtenida para el equipo de climatización. Un análisis detallado demuestra que el control de flujo másico a partir de aprendizaje por refuerzos profundo logra una disminución del 19% de la energía requerida para climatización, manteniendo niveles de confort similares al control convencional utilizado en viviendas de Mendoza.

**Palabras Claves:** Automatización, aprendizaje por refuerzos profundo, estrategias bioclimáticas, optimización energética, confort térmico