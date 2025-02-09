from fluent.runtime import FluentLocalization, FluentResourceLoader

loader = FluentResourceLoader("locale_files/{locale}")

i18n = FluentLocalization(["ru"], ["main.ftl"], loader)
