import * as pxt from '@pixeltable/sdk';
import { string } from '@pixeltable/sdk/functions';

export const TableModel = pxt.modelBase();

export class Films extends TableModel('films', { comment: 'Films and their box office, defined in TypeScript' }) {
  title = pxt.string();
  year = pxt.int();
  budget = pxt.float().orNull().comment('production budget, in USD millions');
  revenue = pxt.float().orNull().comment('worldwide gross, in USD millions');

  // each computed column builds on the columns above it
  title_upper = this.title.upper();
  title_len = this.title_upper.len();
  profit = this.revenue.sub(this.budget);
  margin = this.profit.div(this.budget);
  is_hit = pxt.fn('pixeltable.functions.globals.coalesce')(this.margin, 0).cast(pxt.float()).gt(1);
  decade = this.year.floorDiv(10).mul(10);
  is_recent_hit = this.year.gte(2010).and(this.is_hit);
  label = string.format('{0} ({1})', this.title, this.year);
}

export type Film = pxt.Row<Films>;
export type NewFilm = pxt.Insert<Films>;
