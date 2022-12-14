from django.contrib.auth.models import User
from django.db import models

class Employee(models.Model):

    user = models.OneToOneField(User,on_delete=models.CASCADE)
    avatar = models.ImageField('Аватар', blank=True, null=True, upload_to="static/images/users/",default="static/images/users/default.jpg")
    firstname = models.CharField('Имя', max_length=40, null=True, blank=True, default='Имя')
    lastname = models.CharField('Фамилия', max_length=40, null=True, blank=True, default='Фамилия')
    height = models.PositiveSmallIntegerField('Рост',default='0')
    weight = models.PositiveSmallIntegerField('Вес',default='0')
    age = models.PositiveSmallIntegerField('Возраст',default='0')
    sex = models.CharField("Пол",max_length=15 ,default='муж')
    isSmoking = models.CharField("Курение",max_length=5,default='нет')

    measurements_count = models.PositiveSmallIntegerField('Количество анализов', default='0')

    pulse = models.PositiveSmallIntegerField('Текущий пульс', null=True, default='0')
    list_of_pulse = [0]*5
    sys_pressure = models.PositiveSmallIntegerField('Текущее систолическое давление', null=True, default='0')
    list_of_sys_pressure = [0]*5
    dias_pressure = models.PositiveSmallIntegerField('Текущее диастолическое давление', null=True, default='0')
    list_of_dias_pressure = [0]*5

    # rbc = models.FloatField("Количество эритроцитов", null=False, default='0')
    # pcv = models.FloatField("Объем кровяных клеток", null=False, default='0')
    # mcv = models.FloatField("Средний объём эритроцита", null=False, default='0')
    # mch = models.FloatField("Cреднеклеточный гемоглобин", null=False, default='0')
    # mchc = models.FloatField("Насыщенность эритроцитов гемоглобином", null=False, default='0')
    # rdw = models.FloatField("Распределение эритроцитов по объему", null=False, default='0')
    # tlc = models.FloatField("Количество лейкоцитов", null=False, default='0')
    # plt = models.FloatField("Количество тромбоцитов", null=False, default='0')
