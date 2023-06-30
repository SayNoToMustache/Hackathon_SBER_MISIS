from math import sqrt
import matplotlib.pyplot as plt

class HelpHackathon:

    def __init__(self, data, tiker_name = 'TICKER') -> None:
        self.uniq_ticker_names = data[tiker_name].unique()

    def average_price(self, data, label_encoder, tiker_name = 'TICKER', average_name_open = 'OPEN', average_name_close = 'CLOSE', average_name_low = 'LOW', average_name_high = 'HIGH', is_print = True):
        """Печатает среднюю цену для каждой акции

        
        data - DataFrame

        lable_encoder - LabelEncoder

        tiker_name - название столбца с названиями акций

        average_name - название столбца с ценами
        
        """
        answer = []
        if is_print:
            print('Средняя цена акции: (sum(OPEN) + sum(CLOSE)) + sum(LOW) + sum(HIGH) / 4 / N')

        for name in self.uniq_ticker_names:
            answer.append((data[data[tiker_name] == name][average_name_open].mean()   #Открытие
                           + data[data[tiker_name] == name][average_name_close].mean()#Закрытие
                           + data[data[tiker_name] == name][average_name_low].mean()  #Мин
                           + data[data[tiker_name] == name][average_name_high].mean())#Мак
                           / 4)                                                       #4 значения

        if is_print:
            for i, name in enumerate(self.uniq_ticker_names):
                print(f'{label_encoder.inverse_transform([name])[0]} среднее значение {answer[i]}')
        else:
            return answer


    def standart_deviation(self, data, label_encoder, tiker_name = 'TICKER', average_name_open = 'OPEN', average_name_close = 'CLOSE', average_name_low = 'LOW', average_name_high = 'HIGH'):
        """Печатает среднее отклонение

         data - DataFrame

        lable_encoder - LabelEncoder

        tiker_name - название столбца с названиями акций

        average_name - название столбца с ценами
        """
        mean_ = HelpHackathon.average_price(self, data, label_encoder, tiker_name=tiker_name, average_name_open=average_name_open, 
                                                               average_name_close=average_name_close, average_name_low=average_name_low, average_name_high=average_name_high, 
                                                               is_print=False)
        data['open(x-u) ^ 2'] = data[average_name_open]
        data['close(x-u) ^ 2'] = data[average_name_close]
        data['low(x-u) ^ 2'] = data[average_name_low]
        data['high(x-u) ^ 2'] = data[average_name_high]
        for i, name in enumerate(self.uniq_ticker_names):
            indexes = data.loc[data[tiker_name] == name].index

            data.loc[indexes, 'open(x-u) ^ 2'] -= mean_[i]
            data.loc[indexes, 'open(x-u) ^ 2'] = data.loc[indexes, 'open(x-u) ^ 2'] ** 2
            data.loc[indexes, 'open(x-u) ^ 2'] /= len(indexes)

            data.loc[indexes, 'close(x-u) ^ 2'] -= mean_[i]
            data.loc[indexes, 'close(x-u) ^ 2'] = data.loc[indexes, 'close(x-u) ^ 2'] ** 2
            data.loc[indexes, 'close(x-u) ^ 2'] /= len(indexes)

            data.loc[indexes, 'low(x-u) ^ 2'] -= mean_[i]
            data.loc[indexes, 'low(x-u) ^ 2'] = data.loc[indexes, 'low(x-u) ^ 2'] ** 2
            data.loc[indexes, 'low(x-u) ^ 2'] /= len(indexes)

            data.loc[indexes, 'high(x-u) ^ 2'] -= mean_[i]
            data.loc[indexes, 'high(x-u) ^ 2'] = data.loc[indexes, 'high(x-u) ^ 2'] ** 2
            data.loc[indexes, 'high(x-u) ^ 2'] /= len(indexes)

            sumator = data.loc[indexes, 'open(x-u) ^ 2'].sum() + data.loc[indexes, 'close(x-u) ^ 2'].sum() + data.loc[indexes, 'low(x-u) ^ 2'].sum() + data.loc[indexes, 'high(x-u) ^ 2'].sum()
            answer = sqrt(sumator)

            print(f'Стандартное отклонение {label_encoder.inverse_transform([name])[0]} - {answer}')

        del data['open(x-u) ^ 2'] 
        del data['close(x-u) ^ 2']
        del data['low(x-u) ^ 2']
        del data['high(x-u) ^ 2']


    def sesonial_gr(self, data_load, label_encoder, tiker_name = 'TICKER', column_name = 'OPEN', i = 0):
        """Отобразить информацию по сезонности и тд
        
        i - label ТИКЕРА [0, n-1]
        """

        from statsmodels.tsa.stattools import adfuller
        from statsmodels.graphics.tsaplots import plot_acf


        data = data_load.dropna()

        plt.plot(data[data[tiker_name] == self.uniq_ticker_names[i]].index, data[data[tiker_name] == self.uniq_ticker_names[i]][column_name])
        plt.xlabel('Индекс')
        plt.ylabel(column_name)
        plt.title(f'{label_encoder.inverse_transform([self.uniq_ticker_names[i]])[0]} Зависимость {column_name} от индекса')
        plt.grid(True)
        plt.show()

        # Тест Дики-Фуллера на стационарность
        result = adfuller(data[data[tiker_name] == self.uniq_ticker_names[i]][column_name])
        print('ADF Statistic:', result[0])
        print('p-value:', result[1])
        print(result[4])

        # визуализация ACF
        plot_acf(data[data[tiker_name] == self.uniq_ticker_names[i]][column_name])
        plt.show()


    def moving_average(self, data, label_encoder, tiker_name = 'TICKER', column_name = 'OPEN', window_size = 1):
        """СОздает в data новую колонку SMA в которой будет храниться скользящее среднее сделанное по заданному размеру окна"""

        data['SMA'] = data[column_name]

        for i, name in enumerate(self.uniq_ticker_names):
            indexes = data.loc[data[tiker_name] == name].index
            
            data.loc[indexes, 'SMA'] = data.loc[indexes, column_name].rolling(window=window_size).mean()

    def hackathon_seasonal_decompose(self, data, label_encoder, label_name = 0, tiker_name = 'TICKER', column_name = 'OPEN', period = 12):
        """Декомпозиция временных рядов"""
        from statsmodels.tsa.seasonal import seasonal_decompose

        print(f'Декомпозиция для {label_encoder.inverse_transform([self.uniq_ticker_names[label_name]])}')
        # Декомпозиция временных рядов
        result = seasonal_decompose(data[data[tiker_name] == self.uniq_ticker_names[label_name]][column_name], model='additive', period=period)

        # Визуализация декомпозиции
        result.plot()
        plt.show()