# Generated by Django 4.1.4 on 2022-12-17 07:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0004_auto_20200502_1639'),
    ]

    operations = [
        migrations.AddField(
            model_name='employee',
            name='mch',
            field=models.FloatField(default='0', verbose_name='Cреднеклеточный гемоглобин'),
        ),
        migrations.AddField(
            model_name='employee',
            name='mchc',
            field=models.FloatField(default='0', verbose_name='Насыщенность эритроцитов гемоглобином'),
        ),
        migrations.AddField(
            model_name='employee',
            name='mcv',
            field=models.FloatField(default='0', verbose_name='Средний объём эритроцита'),
        ),
        migrations.AddField(
            model_name='employee',
            name='pcv',
            field=models.FloatField(default='0', verbose_name='Объем кровяных клеток'),
        ),
        migrations.AddField(
            model_name='employee',
            name='plt',
            field=models.FloatField(default='0', verbose_name='Количество тромбоцитов'),
        ),
        migrations.AddField(
            model_name='employee',
            name='rbc',
            field=models.FloatField(default='0', verbose_name='Количество эритроцитов'),
        ),
        migrations.AddField(
            model_name='employee',
            name='rdw',
            field=models.FloatField(default='0', verbose_name='Распределение эритроцитов по объему'),
        ),
        migrations.AddField(
            model_name='employee',
            name='tlc',
            field=models.FloatField(default='0', verbose_name='Количество лейкоцитов'),
        ),
    ]
