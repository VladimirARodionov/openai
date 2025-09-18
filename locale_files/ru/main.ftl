start_menu_text = Привет! Я бот, основанный на Агни Йоге. Задайте мне вопрос.
my_profile_menu = 👤 Мой профиль
my_profile_text = Мой профиль
bot_stopped = Бот остановлен
get_profile_text = 👉 Ваш телеграм ник: {$username}.
   Ваш Телеграм ID: <code><b>{$id}</b></code>
back_menu = <- Назад
back_menu_text = Назад
back_text = Нажато Назад
start_menu = Старт
stop_menu = Стоп
add_user_menu = ✅️ Добавить пользователя
add_user_text = Добавить пользователя
add_user_prompt = Введите id пользователя чтобы его добавить
delete_user_menu = ❎ Удалить пользователя
delete_user_prompt = Введите id пользователя чтобы его удалить
delete_user_text = Удалить пользователя
success_add_user_text = Пользователь добавлен
not_success_add_user_text = Пользователь не найден или уже добавлен
success_delete_user_text = Пользователь удален
not_success_delete_user_text = Пользователь не найден или уже удален
chat_stopped = Чат остановлен
use_menu = Воспользуйтесь меню:
show_users_menu = 📋 Список пользователей
show_users_text = Список пользователей
show_users_text_empty = Список пользователей пуст
toggle_inet_menu = 🔄 Искать в интернете?
toggle_inet_text = Искать в интернете?
user_not_found = Пользователь не найден
toggle_inet_on = Искать в интернете разрешено
toggle_inet_off = Искать в интернете запрещено
response_format_text = Выберите формат ответа:
simple_response_menu = Обычный ответ
detailed_report_menu = 📝 Цитаты+GPT
no_name_text = Нет данных в вопросе. Задайте вопрос заново

# Сообщения для процесса загрузки документов
loading_already_running = Загрузка документов уже выполняется
loading_started = Загрузка документов начата в фоновом режиме
loading_status = 📊 Статус загрузки документов:
loading_total_docs = Всего документов в системе: {$count}
loading_time = Время: {$time}
loading_complete = ✅ Загрузка завершена!
loading_error = ❌ Ошибка при загрузке документов: {$error}
loading_files_count = Загружено {$count} файлов
loading_no_files = Не найдено поддерживаемых файлов в указанной директории

# Сообщения для поиска и отчетов
search_local_title = 🔍 Основной ответ из документов:
search_summary_title = 📝 Краткое саммари локальных документов:
search_internet_title = 🌐 Дополнительная информация из интернета:
search_internet_error = ⚠️ Не удалось получить информацию из интернета
search_sources_title = 📚 Основные локальные источники:
search_source_count = - {$source}: {$count} релевантных фрагментов
search_error = Произошла ошибка: {$error}
search_report_error = Произошла ошибка при формировании отчета: {$error}
search_vector_unavailable = ⚠️ Векторный поиск по локальным документам временно недоступен из-за высокой нагрузки. Используется только интернет-поиск.
search_vector_unavailable_no_fallback = ⚠️ Векторный поиск временно недоступен из-за высокой нагрузки. Попробуйте позже или включите поиск в интернете в настройках.

# Сообщения для работы с базой данных
db_cleared = База данных очищена
db_clear_error = Ошибка при очистке базы данных: {$error}
db_clear_partial = Не все документы были удалены. Осталось: {$count}

# Сообщения для пагинации списка пользователей
show_users_page_header = 📋 Список пользователей (страница {$current} из {$total}, всего: {$count})

# Кнопки навигации по страницам
users_nav_prev = ◀️ Назад
users_nav_next = Вперед ▶️
users_nav_back_to_menu = 🔙 В главное меню

# Сообщения для обработки запросов
processing_message = ⏳ Подготавливаю ответ...
error_processing_request = Произошла ошибка при обработке запроса: {$error}
error_in_callback = Произошла ошибка: {$error}
